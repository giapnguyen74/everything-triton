"""03 - Flash Attention (forward pass).

Builds directly on 02's fused softmax. The problem: a full attention row is
`q @ Kᵀ` over the whole sequence, which is far too wide to load into SRAM at
once (and we also have V to multiply in). The fix is the *online* softmax from
02's docstring: stream K/V in blocks, carrying a running max `m`, running
denominator `l`, and unnormalized output accumulator `õ`, rescaling the old
state by `exp(m - m_new)` whenever a bigger value shows up.

Two axes, two different roles:
  - Q (queries)  -> PARALLEL. Rows are independent, so each query block is its
                    own program in the grid. No combining across them.
  - K/V          -> SEQUENTIAL inner loop. The softmax reduces over this axis,
                    so each program streams all of K/V past its resident Q
                    block, updating the online state block by block.

Each program holds one BLOCK_M-row tile of Q in SRAM and streams K/V through it
-> K/V loads are amortized across all BLOCK_M queries, and the matmuls are real
GEMMs (tensor cores) instead of per-row GEMVs.

Per query row the recurrence is exactly 02's, just elementwise across the tile:

    init:   m = -inf,  l = 0,  õ = 0
    block:  m⁺ = max(m, max(s_block))
            α  = exp(m - m⁺)
            l  = α·l + Σ exp(s_block - m⁺)
            õ  = α·õ + exp(s_block - m⁺) @ v_block
            m  = m⁺
    final:  o = õ / l

Exact, not an approximation: the α factor rebases earlier blocks onto the new
max perfectly. This forward kernel is non-causal; causal masking is the natural
next step (skip key blocks past the query, mask the diagonal block).

    ./run_interpret.sh 03_flash_attention.py   # CPU logic check (slow)
    ./run_gpu.sh       03_flash_attention.py   # GPU run + benchmark
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, INTERPRET, banner


@triton.jit
def flash_kernel(q_ptr, k_ptr, v_ptr, out_ptr, scale, n_ctx,
                 HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)        # which query block
    pid_z = tl.program_id(1)        # which (batch * head)

    base = pid_z * n_ctx * HEAD_DIM
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < n_ctx

    # Load this program's Q tile once; it stays resident while K/V stream past.
    q_ptrs = q_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

    # Online-softmax running state, one entry per query row in the tile.
    m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    # Inner loop: stream K/V blocks, updating the running state.
    for start_n in range(0, n_ctx, BLOCK_N):
        cur_n = start_n + offs_n
        n_mask = cur_n < n_ctx

        k_ptrs = k_ptr + base + cur_n[:, None] * HEAD_DIM + offs_d[None, :]
        v_ptrs = v_ptr + base + cur_n[:, None] * HEAD_DIM + offs_d[None, :]
        k = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0)

        s = tl.dot(q, tl.trans(k)) * scale            # (BLOCK_M, BLOCK_N)
        s = tl.where(n_mask[None, :], s, -float("inf"))  # ignore padded keys

        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)                   # rescale factor for old state
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]                          # deferred normalization

    out_ptrs = out_ptr + base + offs_m[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None])


def flash_attention(q, k, v, BLOCK_M=64, BLOCK_N=64):
    # q, k, v: (batch, heads, seq, head_dim)
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)
    q, k, v = (t.reshape(B * H, N, D).contiguous() for t in (q, k, v))
    out = torch.empty_like(q)
    grid = (triton.cdiv(N, BLOCK_M), B * H)           # parallel over (Q blocks, batch*heads)
    flash_kernel[grid](q, k, v, out, scale, N,
                       HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return out.reshape(B, H, N, D)


def ref_attention(q, k, v):
    scale = 1.0 / (q.shape[-1] ** 0.5)
    p = torch.softmax((q @ k.transpose(-2, -1)) * scale, dim=-1)
    return p @ v


def main():
    banner()
    B, H, N, D = 2, 8, 1024, 64
    q = torch.randn(B, H, N, D, device=DEVICE)
    k = torch.randn(B, H, N, D, device=DEVICE)
    v = torch.randn(B, H, N, D, device=DEVICE)

    torch.testing.assert_close(flash_attention(q, k, v), ref_attention(q, k, v),
                               atol=1e-2, rtol=0)
    print("flash attention: correct ✓")

    if not INTERPRET:
        import torch.nn.functional as F
        print("triton:", triton.testing.do_bench(lambda: flash_attention(q, k, v)), "ms")
        print("torch :", triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(q, k, v)), "ms")


if __name__ == "__main__":
    main()
