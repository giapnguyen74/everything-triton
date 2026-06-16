#!/usr/bin/env bash
# Fast logic/correctness loop: pure-Python interpreter on CPU, no GPU needed.
# Real print()/pdb work. Usage: ./run_interpret.sh 01_vector_add.py
set -euo pipefail
cd "$(dirname "$0")"
TRITON_INTERPRET=1 python "$@"
