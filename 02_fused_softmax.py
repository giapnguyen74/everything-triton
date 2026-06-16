"""02 - Fused row-wise softmax.

The conceptual seed of Flash Attention: numerically-stable softmax
(subtract the row max, then normalize). Each program loads a whole row into
SRAM and does max -> exp -> sum -> divide without round-tripping to HBM.

Flash Attention generalizes this to the *online* version: a running max and
running denominator updated block by block, so the full row never has to fit
in SRAM at once.

    ./run_interpret.sh 02_fused_softmax.py
    ./run_gpu.sh       02_fused_softmax.py
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, INTERPRET, banner


@triton.jit
def softmax_kernel(out_ptr, in_ptr, in_row_stride, out_row_stride,
                   n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
    x = x - tl.max(x, axis=0)            # numerically stable
    num = tl.exp(x)
    out = num / tl.sum(num, axis=0)
    tl.store(out_ptr + row * out_row_stride + cols, out, mask=mask)


def softmax(x):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(n_cols)
    softmax_kernel[(n_rows,)](out, x, x.stride(0), out.stride(0), n_cols, BLOCK=BLOCK)
    return out


def main():
    banner()
    x = torch.randn(1823, 781, device=DEVICE)

    torch.testing.assert_close(softmax(x), torch.softmax(x, axis=1), atol=1e-5, rtol=0)
    print("fused softmax: correct ✓")

    if not INTERPRET:
        print("triton:", triton.testing.do_bench(lambda: softmax(x)), "ms")
        print("torch :", triton.testing.do_bench(lambda: torch.softmax(x, axis=1)), "ms")


if __name__ == "__main__":
    main()
