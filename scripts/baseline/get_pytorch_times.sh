#!/bin/bash
#SBATCH --job-name=mlir-pytorch-times
#SBATCH --partition=compute
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/get_pytorch_times_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/get_pytorch_times_%j.err
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch scripts/get_pytorch_times.sh <config>
#   sbatch scripts/get_pytorch_times.sh <config> [implementation]
#
# Examples:
#   sbatch scripts/get_pytorch_times.sh config/old_dataset/train/train1.json
#
# All paths (benchmarks dir, output JSON) are derived from the config file.
# Override individual paths with --benchmarks-dir / --output if needed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

CONFIG="${1:?Usage: sbatch scripts/get_pytorch_times.sh <config>}"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi

CONFIG_IMPL=$(python3 - <<PY
import json
try:
    with open("$CONFIG", "r") as f:
        print((json.load(f).get("implementation") or "").strip())
except Exception:
    print("")
PY
)

IMPLEMENTATION="${2:-${CONFIG_IMPL:-${AUTOSCHEDULER_IMPL:-rl_autoschedular_v0}}}"
export AUTOSCHEDULER_IMPL="$IMPLEMENTATION"
export CONFIG_FILE_PATH="$CONFIG"

echo "=========================================="
echo "get_pytorch_times.py started at $(date)"
echo "Config: $CONFIG"
echo "Implementation: $AUTOSCHEDULER_IMPL"
echo "Node:   $(hostname)"
echo "=========================================="

CHUNK_IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_CHUNKS=${SLURM_ARRAY_TASK_COUNT:-1}

python scripts/baseline/get_pytorch_times.py --config "$CONFIG" --chunk-index $CHUNK_IDX --num-chunks $NUM_CHUNKS

echo "=========================================="
echo "get_pytorch_times.py completed at $(date)"
echo "=========================================="
