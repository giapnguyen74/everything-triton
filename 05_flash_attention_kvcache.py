"""05 - Flash Attention with a KV cache (offset causal), PyTorch vs Triton.

The inference case: queries are the *last* `q_len` rows of a longer `total_k`
sequence (the rest of K/V come from the cache). The causal rule gains an offset:

    query i sits at absolute position  (total_k - q_len) + i
    key   j sits at absolute position  j
    attend iff  key_pos <= query_pos   ->   j <= offset + i,  offset = total_k - q_len

This single rule covers all three regimes the PyTorch code special-cases:
    q_len == total_k        -> offset 0          -> plain causal
    1 < q_len < total_k     -> offset > 0         -> chunked / ragged prefill
    q_len == 1  (decode)    -> offset total_k-1   -> the one query sees every key

`torch_attention` below is the reference (exactly the snippet being ported).
`flash_attention_kv` is the Triton kernel: same online-softmax forward as 03/04,
but the causal mask and loop bound use the *absolute* query position
`offset + offs_m`, and Q vs K/V now have different lengths.

Forward only (inference); for training grads use 04.

    ./run_gpu.sh 05_flash_attention_kvcache.py
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from common import DEVICE, INTERPRET, banner


# --------------------------------------------------------------------------- #
# PyTorch reference (the code being ported)
# --------------------------------------------------------------------------- #
def torch_attention(q, k, v):
    q_len, total_k = q.shape[-2], k.shape[-2]
    if q_len > 1 and q_len != total_k:
        qpos = torch.arange(total_k - q_len, total_k, device=q.device)
        kpos = torch.arange(total_k, device=q.device)
        mask = (kpos[None, :] <= qpos[:, None])[None, None]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    else:
        return F.scaled_dot_product_attention(q, k, v, is_causal=q_len > 1)


# --------------------------------------------------------------------------- #
# Triton offset-causal forward
# --------------------------------------------------------------------------- #
def _configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm in (64, 128) for bn in (32, 64) for w in (4, 8) for s in (2, 3)
    ]


@triton.autotune(configs=_configs(), key=["n_ctx_q", "n_ctx_k", "HEAD_DIM"])
@triton.jit
def kv_fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, scale, n_ctx_q, n_ctx_k,
                  HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    q_base = pid_z * n_ctx_q * HEAD_DIM          # Q has q_len rows
    kv_base = pid_z * n_ctx_k * HEAD_DIM         # K/V have total_k rows
    offset = n_ctx_k - n_ctx_q                   # the KV-cache offset

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    q_pos = offset + offs_m                       # absolute position of each query
    q_mask = offs_m < n_ctx_q                     # decode: q_len may be < BLOCK_M

    q = tl.load(q_ptr + q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
                mask=q_mask[:, None], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # A query at absolute pos q_pos can see keys up to q_pos; the block's max
    # query is offset + (pid_m+1)*BLOCK_M - 1, so stop the loop there.
    hi = tl.minimum(offset + (pid_m + 1) * BLOCK_M, n_ctx_k)
    for start_n in range(0, hi, BLOCK_N):
        cur_n = start_n + tl.arange(0, BLOCK_N)
        k_mask = cur_n < n_ctx_k
        k = tl.load(k_ptr + kv_base + cur_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=k_mask[:, None], other=0.0)
        v = tl.load(v_ptr + kv_base + cur_n[:, None] * HEAD_DIM + offs_d[None, :],
                    mask=k_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        keep = (q_pos[:, None] >= cur_n[None, :]) & k_mask[None, :]   # offset causal
        s = tl.where(keep, s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_ptr + q_base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(o_ptr.dtype.element_ty), mask=q_mask[:, None])


def flash_attention_kv(q, k, v):
    # q: (B,H,q_len,D)   k,v: (B,H,total_k,D)   total_k >= q_len
    B, H, Sq, D = q.shape
    Sk = k.shape[-2]
    scale = 1.0 / (D ** 0.5)
    q = q.reshape(B * H, Sq, D).contiguous()
    k = k.reshape(B * H, Sk, D).contiguous()
    v = v.reshape(B * H, Sk, D).contiguous()
    o = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(Sq, meta["BLOCK_M"]), B * H)
    kv_fwd_kernel[grid](q, k, v, o, scale, Sq, Sk, HEAD_DIM=D)
    return o.reshape(B, H, Sq, D)


# --------------------------------------------------------------------------- #
# Compare
# --------------------------------------------------------------------------- #
def main():
    banner()
    B, H, D = 2, 8, 64
    cases = [
        ("full causal",      1024, 1024),
        ("chunked prefill",   128, 1024),
        ("ragged prefill",    200, 1000),
        ("decode (q_len=1)",    1, 1024),
    ]
    for tag, Sq, Sk in cases:
        q = torch.randn(B, H, Sq, D, device=DEVICE, dtype=torch.bfloat16)
        k = torch.randn(B, H, Sk, D, device=DEVICE, dtype=torch.bfloat16)
        v = torch.randn(B, H, Sk, D, device=DEVICE, dtype=torch.bfloat16)

        out = flash_attention_kv(q, k, v)
        ref = torch_attention(q, k, v)
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)
        line = f"{tag:18s} Sq={Sq:<5d} Sk={Sk:<5d} correct ✓"

        if not INTERPRET:
            t = triton.testing.do_bench(lambda: flash_attention_kv(q, k, v))
            r = triton.testing.do_bench(lambda: torch_attention(q, k, v))
            line += f"   triton {t:.4f} ms   torch {r:.4f} ms"
        print(line)


if __name__ == "__main__":
    main()
