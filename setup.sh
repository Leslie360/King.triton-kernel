#!/bin/bash
# King.triton-kernel 安装脚本 (参考 Dr.Kernel setup.sh)
set -euo pipefail
echo "=== King.triton-kernel setup ==="
# conda env 建议: drkernel (python3.10)
pip install -r requirements.txt
echo "✅ deps 安装完成"
