#!/usr/bin/env bash
# Wrapper to run import and compile checks for torch-mlir
set -euo pipefail

echo "== Quick import test =="
python - <<'PY'
import sys
try:
    import torch, torch_mlir
    print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
    print("torch-mlir import OK ->", torch_mlir.__file__)
except Exception as e:
    print("Import failed:", e)
    sys.exit(1)
PY

echo
echo "== Functional compile test =="
python scripts/utils/test_torch_mlir_compile.py && echo "Compile test PASS" || { echo "Compile test FAIL"; exit 1; }

echo "All torch-mlir checks passed."
