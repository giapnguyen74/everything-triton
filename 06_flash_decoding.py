"""06 - Flash-Decoding: split-KV attention for the q_len == 1 decode step.

05's decode row was slow because a single query gives almost no parallelism:
the grid is just (1, batch*heads) and each program serially scans the whole
K/V cache. Decode is memory-bound (read the entire cache, do one query's worth
of compute), so the fix is to read the cache *in parallel*.

Split-KV (a.k.a. Flash-Decoding) parallelizes the reduction axis:

  kernel 1 (split):   partition the cache into NUM_SPLITS chunks and run one
                      program per (chunk x head x batch). Each streams its chunk
                      with the usual online softmax and writes a PARTIAL result
                      o_c (unnormalized), plus its local max m_c and denom l_c.
  kernel 2 (combine): merge the NUM_SPLITS partials with the same online rescale
                      used inside the inner loop -- exp(m_c - m_global) -- and
                      normalize once.

Merging chunk-partials is mathematically identical to merging block-partials in
03/04; the only difference is the partials live in HBM (written by separate
programs) instead of registers. The decode query attends to the whole cache, so
there is no causal mask here. The matmuls are GEMV (one query), so tensor cores
don't apply -- this is purely about saturating memory bandwidth.

    ./run_gpu.sh 06_flash_decoding.py
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import DEVICE, INTERPRET, banner


@triton.jit
def split_kernel(q_ptr, k_ptr, v_ptr, po_ptr, pm_ptr, pl_ptr, scale, n_ctx_k,
                 HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_s = tl.program_id(0)          # which cache chunk
    pid_z = tl.program_id(1)          # which (batch * head)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + pid_z * HEAD_DIM + offs_d).to(tl.float32)   # single query vector

    chunk = tl.cdiv(n_ctx_k, NUM_SPLITS)
    lo = pid_s * chunk
    hi = tl.minimum(lo + chunk, n_ctx_k)
    kv_base = pid_z * n_ctx_k * HEAD_DIM

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)

    for start_n in range(lo, hi, BLOCK_N):
        cur_n = start_n + tl.arange(0, BLOCK_N)
        mask = cur_n < hi
        k = tl.load(k_ptr + kv_base + cur_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)
        v = tl.load(v_ptr + kv_base + cur_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=mask[:, None], other=0.0).to(tl.float32)

        s = tl.sum(q[None, :] * k, axis=1) * scale          # (BLOCK_N,) GEMV
        s = tl.where(mask, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    # Write this chunk's partials (unnormalized output + max + denominator).
    po_off = (pid_z * NUM_SPLITS + pid_s) * HEAD_DIM
    tl.store(po_ptr + po_off + offs_d, acc)
    tl.store(pm_ptr + pid_z * NUM_SPLITS + pid_s, m_i)
    tl.store(pl_ptr + pid_z * NUM_SPLITS + pid_s, l_i)


@triton.jit
def combine_kernel(po_ptr, pm_ptr, pl_ptr, o_ptr,
                   HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr):
    pid_z = tl.program_id(0)
    offs_s = tl.arange(0, NUM_SPLITS)
    offs_d = tl.arange(0, HEAD_DIM)

    m_c = tl.load(pm_ptr + pid_z * NUM_SPLITS + offs_s)        # (NUM_SPLITS,)
    l_c = tl.load(pl_ptr + pid_z * NUM_SPLITS + offs_s)
    m = tl.max(m_c, axis=0)                                    # global max
    sc = tl.exp(m_c - m)                                       # rebase each chunk
    l = tl.sum(sc * l_c, axis=0)

    po = tl.load(po_ptr + (pid_z * NUM_SPLITS + offs_s)[:, None] * HEAD_DIM
                 + offs_d[None, :])                            # (NUM_SPLITS, HEAD_DIM)
    o = tl.sum(sc[:, None] * po, axis=0) / l
    tl.store(o_ptr + pid_z * HEAD_DIM + offs_d, o.to(o_ptr.dtype.element_ty))


def pick_splits(Sk, Z, BLOCK_N=64, target_programs=256):
    """Choose num_splits to fill the GPU: enough programs to saturate the SMs,
    but no finer than one block per split. Rounded down to a power of two
    (combine_kernel indexes the splits with tl.arange)."""
    max_useful = max(1, Sk // BLOCK_N)            # don't split finer than a block
    by_occupancy = max(1, target_programs // Z)   # enough programs to fill the SMs
    n = min(by_occupancy, max_useful)
    return 1 << (n.bit_length() - 1)              # round down to power of two


def flash_decode(q, k, v, num_splits=None, BLOCK_N=64):
    # q: (B,H,1,D)   k,v: (B,H,Sk,D)
    B, H, _, D = q.shape
    Sk = k.shape[-2]
    scale = 1.0 / (D ** 0.5)
    Z = B * H
    if num_splits is None:
        num_splits = pick_splits(Sk, Z, BLOCK_N)
    q = q.reshape(Z, D).contiguous()
    k = k.reshape(Z, Sk, D).contiguous()
    v = v.reshape(Z, Sk, D).contiguous()

    po = torch.empty((Z, num_splits, D), dtype=torch.float32, device=q.device)
    pm = torch.empty((Z, num_splits), dtype=torch.float32, device=q.device)
    pl = torch.empty((Z, num_splits), dtype=torch.float32, device=q.device)

    split_kernel[(num_splits, Z)](q, k, v, po, pm, pl, scale, Sk,
                                  HEAD_DIM=D, NUM_SPLITS=num_splits, BLOCK_N=BLOCK_N)
    o = torch.empty((Z, D), dtype=k.dtype, device=q.device)
    combine_kernel[(Z,)](po, pm, pl, o, HEAD_DIM=D, NUM_SPLITS=num_splits)
    return o.reshape(B, H, 1, D)


def main():
    banner()
    B, H, D = 2, 8, 64
    for Sk in (1024, 4096, 16384):
        q = torch.randn(B, H, 1, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, Sk, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, H, Sk, D, device=DEVICE, dtype=torch.bfloat16)

        splits = pick_splits(Sk, B * H)
        out = flash_decode(q, k, v)
        ref = F.scaled_dot_product_attention(q, k, v)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)
        line = f"Sk={Sk:<6d} splits={splits:<3d} correct ✓"

        if not INTERPRET:
            fd = triton.testing.do_bench(lambda: flash_decode(q, k, v))
            r = triton.testing.do_bench(lambda: F.scaled_dot_product_attention(q, k, v))
            line += f"   flash_decode {fd:.4f} ms   torch {r:.4f} ms"
        print(line)


if __name__ == "__main__":
    main()
