# everything-triton
triton learning

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
