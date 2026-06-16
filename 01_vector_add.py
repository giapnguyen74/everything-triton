"""01 - Vector add: the hello-world of Triton.

One program instance handles one BLOCK-sized chunk. `mask` guards the tail
when the length isn't a multiple of BLOCK.

    ./run_interpret.sh 01_vector_add.py   # CPU, fast logic check
    ./run_gpu.sh       01_vector_add.py   # GPU, real run + benchmark
"""
import torch
import triton
import triton.language as tl

from common import DEVICE, INTERPRET, banner


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


def add(x, y):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    add_kernel[grid](x, y, out, n, BLOCK=1024)
    return out


def main():
    banner()
    x = torch.randn(98_432, device=DEVICE)
    y = torch.randn(98_432, device=DEVICE)

    torch.testing.assert_close(add(x, y), x + y)
    print("vector add: correct ✓")

    if not INTERPRET:  # benchmarking is meaningless on the CPU interpreter
        print("triton:", triton.testing.do_bench(lambda: add(x, y)), "ms")
        print("torch :", triton.testing.do_bench(lambda: x + y), "ms")


if __name__ == "__main__":
    main()
