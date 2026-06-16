# everything-triton

A hands-on walk from the Triton "hello world" up to a complete Flash Attention
with a backward pass. Each file is self-contained and checks itself against a
PyTorch reference, then benchmarks against it.

## The progression

| File | Teaches |
|------|---------|
| `01_vector_add.py` | Program/grid model: one program handles one BLOCK-sized chunk, `mask` guards the tail. |
| `02_fused_softmax.py` | Fusing many ops (max → exp → sum → divide) into one kernel; numerically-stable, one row per program. The seed of Flash Attention. |
| `03_flash_attention.py` | The online-softmax trick: stream K/V in blocks with a running max/denominator so a full row never has to fit in SRAM. Minimal fp32 forward. |
| `04_flash_attention_full.py` | The real thing: bf16 + `@triton.autotune`, causal masking, and a backward pass that recomputes P from the saved logsumexp instead of storing the N×N matrix. Wrapped in a `torch.autograd.Function`. |

The core idea carried from `02` onward: the softmax reduces over the K/V axis, so
K/V is streamed (sequential inner loop with online state) while queries/heads run
in parallel across the grid. Bigger query tiles amortize K/V loads and turn the
matmuls into tensor-core GEMMs.

## Running

```bash
./run_interpret.sh 01_vector_add.py   # CPU logic check, no GPU needed
./run_gpu.sh       01_vector_add.py   # real GPU run + benchmark
```

## Benchmarks (NVIDIA GB10, torch 2.12.0+cu130, triton 3.7.0)

Times are per call; `torch` is the PyTorch reference for that op.

| Kernel | triton | torch | notes |
|--------|--------|-------|-------|
| `01` vector add (98k) | 0.0151 ms | 0.0133 ms | memory-bound; torch's single add is hard to beat |
| `02` fused softmax (1823×781) | 0.0841 ms | 0.0849 ms | fusion edges ahead of torch's multi-kernel softmax |
| `03` flash attn fwd, fp32 (2×8×1024×64) | 0.219 ms | 0.406 ms | torch SDPA falls back to fp32 math (no flash); we use TF32 |
| `04` flash attn fwd, bf16, non-causal | 0.0907 ms | 0.0911 ms | parity with torch's real bf16 flash kernel |
| `04` flash attn fwd, bf16, causal | 0.0697 ms | 0.0997 ms | ~1.4× faster; skipping the upper triangle pays off |

`04` also verifies the **backward** pass (dQ/dK/dV) against autograd for both
causal and non-causal. The algorithm was cross-checked with a NumPy port of the
exact tiled forward+backward against analytic gradients (errors ~1e-15).

## Troubleshooting

### `subprocess.CalledProcessError` from gcc when initializing the CUDA driver

Triton compiles a small helper (`cuda_utils.c`) at runtime. If this gcc step
fails, you'll see a `CalledProcessError` from `triton/runtime/build.py`. Triton
silences stdout but not stderr, so re-run the failing gcc command manually to
see the real error.

Common causes:

- **Missing Python development headers** (`Python.h` not found). Install the
  matching `python-dev` package for your interpreter version, e.g.:

  ```bash
  sudo apt install python3.12-dev   # match your Python version (pythonXX-dev)
  ```

- **`libcuda.so.1` not on the linker path** — check with
  `ldconfig -p | grep libcuda` and add its directory to `LIBRARY_PATH` if needed
  (on Jetson it's often `/usr/lib/aarch64-linux-gnu/tegra`).

- **Missing CUDA headers** (`cuda.h`) under
  `triton/backends/nvidia/include/`.
