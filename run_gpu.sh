#!/usr/bin/env bash
# Real run: compiles to the GPU and benchmarks. Usage: ./run_gpu.sh 01_vector_add.py
set -euo pipefail
cd "$(dirname "$0")"
unset TRITON_INTERPRET
python "$@"
