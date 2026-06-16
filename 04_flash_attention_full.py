"""04 - Flash Attention, the full version.

Picks up where 03 left off and adds the three things a real kernel needs:

  1. bf16 + autotune   - inputs in bf16 (so torch's own flash kernel becomes the
                         baseline, not the 02-style multi-kernel strawman), fp32
                         accumulation, and @triton.autotune to pick the tile.
  2. Causal masking    - a query may only attend to keys at or before it. We skip
                         K/V blocks entirely past the diagonal and mask the
                         diagonal block, ~halving the work.
  3. Backward pass     - gradients via softmax *recomputation* (the flash trick:
                         never store the N×N matrix, rebuild P from the saved
                         logsumexp L). Wrapped in a torch.autograd.Function so it
                         plugs into normal autograd.

Backward math (P = softmax(scale·QKᵀ), O = P·V):
    delta_i = Σ_k O_ik · dO_ik          (one scalar per query row)
    dV = Pᵀ · dO
    dP = dO · Vᵀ
    dS = P ⊙ (dP - delta)               (softmax Jacobian, elementwise)
    dQ = scale · dS · K
    dK = scale · dSᵀ · Q
We never materialize P or dS in HBM: each backward program recomputes its tile
of P from Q, K and the saved L = m + log(l), exactly as in the forward.

Parallelism mirrors the forward's logic:
  - dK/dV : one program per K/V block, loops over Q blocks (reduction over Q).
  - dQ    : one program per Q block,  loops over K/V blocks (reduction over K/V).

Assumes seq len is a multiple of the tile and head_dim is a power of two
(true for the test below); add padding masks like 03 for ragged shapes.

    ./run_gpu.sh 04_flash_attention_full.py
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, INTERPRET, banner


# --------------------------------------------------------------------------- #
# Forward
# --------------------------------------------------------------------------- #
def _fwd_configs():
    return [
        triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
        for bm in (64, 128) for bn in (32, 64) for w in (4, 8) for s in (2, 3)
    ]


@triton.autotune(configs=_fwd_configs(), key=["n_ctx", "HEAD_DIM", "CAUSAL"])
@triton.jit
def fwd_kernel(q_ptr, k_ptr, v_ptr, o_ptr, L_ptr, scale, n_ctx,
               HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    base = pid_z * n_ctx * HEAD_DIM
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n0 = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])

    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Causal: keys past this query block can't contribute, so stop the loop early.
    hi = (pid_m + 1) * BLOCK_M if CAUSAL else n_ctx
    for start_n in range(0, hi, BLOCK_N):
        cur_n = start_n + offs_n0
        k = tl.load(k_ptr + base + cur_n[:, None] * HEAD_DIM + offs_d[None, :])
        v = tl.load(v_ptr + base + cur_n[:, None] * HEAD_DIM + offs_d[None, :])

        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        if CAUSAL:
            s = tl.where(offs_m[:, None] >= cur_n[None, :], s, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(o_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             acc.to(o_ptr.dtype.element_ty))
    # Save logsumexp so the backward can rebuild P without storing it.
    tl.store(L_ptr + pid_z * n_ctx + offs_m, m_i + tl.log(l_i))


# --------------------------------------------------------------------------- #
# Backward
# --------------------------------------------------------------------------- #
@triton.jit
def bwd_preprocess(o_ptr, do_ptr, delta_ptr, n_ctx,
                   HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    base = pid_z * n_ctx * HEAD_DIM
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    o = tl.load(o_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]).to(tl.float32)
    do = tl.load(do_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]).to(tl.float32)
    tl.store(delta_ptr + pid_z * n_ctx + offs_m, tl.sum(o * do, axis=1))


@triton.jit
def bwd_dkdv_kernel(q_ptr, k_ptr, v_ptr, do_ptr, L_ptr, delta_ptr, dk_ptr, dv_ptr,
                    scale, n_ctx, HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
                    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # One program per K/V block; reduces over query blocks.
    pid_n = tl.program_id(0)
    pid_z = tl.program_id(1)
    base = pid_z * n_ctx * HEAD_DIM
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    k = tl.load(k_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :])
    v = tl.load(v_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :])
    dk = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK_N, HEAD_DIM), dtype=tl.float32)

    # Causal: only queries at or after this key block contribute.
    lo = (pid_n * BLOCK_N // BLOCK_M) * BLOCK_M if CAUSAL else 0
    for start_m in range(lo, n_ctx, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q = tl.load(q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
        do = tl.load(do_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
        L = tl.load(L_ptr + pid_z * n_ctx + offs_m)
        delta = tl.load(delta_ptr + pid_z * n_ctx + offs_m)

        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        p = tl.exp(s - L[:, None])                       # recompute P tile
        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        dv += tl.dot(tl.trans(p).to(do.dtype), do).to(tl.float32)
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = scale * p * (dp - delta[:, None])
        dk += tl.dot(tl.trans(ds).to(q.dtype), q).to(tl.float32)

    tl.store(dk_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
             dk.to(dk_ptr.dtype.element_ty))
    tl.store(dv_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :],
             dv.to(dv_ptr.dtype.element_ty))


@triton.jit
def bwd_dq_kernel(q_ptr, k_ptr, v_ptr, do_ptr, L_ptr, delta_ptr, dq_ptr,
                  scale, n_ctx, HEAD_DIM: tl.constexpr, CAUSAL: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # One program per Q block; reduces over K/V blocks.
    pid_m = tl.program_id(0)
    pid_z = tl.program_id(1)
    base = pid_z * n_ctx * HEAD_DIM
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
    do = tl.load(do_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :])
    L = tl.load(L_ptr + pid_z * n_ctx + offs_m)
    delta = tl.load(delta_ptr + pid_z * n_ctx + offs_m)
    dq = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    hi = (pid_m + 1) * BLOCK_M if CAUSAL else n_ctx
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        k = tl.load(k_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :])
        v = tl.load(v_ptr + base + offs_n[:, None] * HEAD_DIM + offs_d[None, :])

        s = tl.dot(q, tl.trans(k)).to(tl.float32) * scale
        p = tl.exp(s - L[:, None])
        if CAUSAL:
            p = tl.where(offs_m[:, None] >= offs_n[None, :], p, 0.0)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = scale * p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k).to(tl.float32)

    tl.store(dq_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :],
             dq.to(dq_ptr.dtype.element_ty))


# --------------------------------------------------------------------------- #
# autograd.Function glue
# --------------------------------------------------------------------------- #
class FlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal):
        B, H, N, D = q.shape
        scale = 1.0 / (D ** 0.5)
        q, k, v = (t.reshape(B * H, N, D).contiguous() for t in (q, k, v))
        o = torch.empty_like(q)
        L = torch.empty((B * H, N), dtype=torch.float32, device=q.device)
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_M"]), B * H)
        fwd_kernel[grid](q, k, v, o, L, scale, N, HEAD_DIM=D, CAUSAL=causal)
        ctx.save_for_backward(q, k, v, o, L)
        ctx.scale, ctx.causal, ctx.shape = scale, causal, (B, H, N, D)
        return o.reshape(B, H, N, D)

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, L = ctx.saved_tensors
        B, H, N, D = ctx.shape
        Z, causal, scale = B * H, ctx.causal, ctx.scale
        do = do.reshape(Z, N, D).contiguous()
        delta = torch.empty((Z, N), dtype=torch.float32, device=q.device)
        dq, dk, dv = (torch.empty_like(t) for t in (q, k, v))
        BM = BN = 64

        bwd_preprocess[(triton.cdiv(N, BM), Z)](o, do, delta, N, HEAD_DIM=D, BLOCK_M=BM)
        bwd_dkdv_kernel[(triton.cdiv(N, BN), Z)](
            q, k, v, do, L, delta, dk, dv, scale, N,
            HEAD_DIM=D, CAUSAL=causal, BLOCK_M=BM, BLOCK_N=BN)
        bwd_dq_kernel[(triton.cdiv(N, BM), Z)](
            q, k, v, do, L, delta, dq, scale, N,
            HEAD_DIM=D, CAUSAL=causal, BLOCK_M=BM, BLOCK_N=BN)

        r = lambda t: t.reshape(B, H, N, D)
        return r(dq), r(dk), r(dv), None


def flash_attention(q, k, v, causal=False):
    return FlashAttention.apply(q, k, v, causal)


# --------------------------------------------------------------------------- #
# Test + benchmark
# --------------------------------------------------------------------------- #
def main():
    banner()
    import torch.nn.functional as F
    B, H, N, D = 2, 8, 1024, 64

    for causal in (False, True):
        q, k, v = (torch.randn(B, H, N, D, device=DEVICE, dtype=torch.bfloat16,
                               requires_grad=True) for _ in range(3))
        dout = torch.randn_like(q)

        out = flash_attention(q, k, v, causal=causal)
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=causal)

        # forward
        torch.testing.assert_close(out, ref, atol=2e-2, rtol=0)
        # backward
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout, retain_graph=True)
        rq, rk, rv = torch.autograd.grad(ref, (q, k, v), dout)
        for a, b in ((dq, rq), (dk, rk), (dv, rv)):
            torch.testing.assert_close(a, b, atol=2e-2, rtol=0)
        print(f"causal={causal}: forward + backward correct ✓")

        if not INTERPRET:
            fwd = triton.testing.do_bench(lambda: flash_attention(q, k, v, causal))
            ref_fwd = triton.testing.do_bench(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal))
            print(f"           fwd  triton {fwd:.4f} ms   torch {ref_fwd:.4f} ms")


if __name__ == "__main__":
    main()
