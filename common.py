"""Shared helpers for the Triton exercises.

Picks the right device automatically:
- TRITON_INTERPRET=1  -> run on CPU (interpreter mode, no GPU needed)
- otherwise           -> run on the GPU ('cuda')

If interpret mode ever complains about CPU tensors on your Triton version,
flip INTERPRET_DEVICE to 'cuda' (older Triton emulated on CPU but still wanted
cuda-allocated tensors).
"""
import os

import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
INTERPRET_DEVICE = "cpu"

if INTERPRET:
    DEVICE = INTERPRET_DEVICE
else:
    assert torch.cuda.is_available(), (
        "No GPU. Run on the workstation, or use ./run_interpret.sh for CPU mode."
    )
    DEVICE = "cuda"


def banner():
    import triton
    mode = "INTERPRET (CPU)" if INTERPRET else f"GPU ({torch.cuda.get_device_name(0)})"
    print(f"[mode] {mode}  |  torch {torch.__version__}  triton {triton.__version__}")
