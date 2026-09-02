#!/bin/bash
# King.triton-kernel install script (adapted from Dr.Kernel setup.sh)
set -euo pipefail
echo "=== King.triton-kernel setup ==="
# Suggested conda env: drkernel (python3.10)
pip install -r requirements.txt
echo "✅ dependencies installed"
