#!/bin/bash
#SBATCH --job-name=mlir-train
#SBATCH --partition=compute
#SBATCH --mem=8G
#SBATCH --cpus-per-task=28
#SBATCH --constraint=bergamo
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/train_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/train_%j.err
#SBATCH --mail-type=END,FAIL
set -e
trap 'echo "TRAINING FAILED"' ERR
#
# Usage:
#   sbatch scripts/train/train.sh                          # uses config/old_dataset/train/train1.json (default)
#   sbatch scripts/train/train.sh config/my_config.json   # uses a custom config
#   sbatch scripts/train/train.sh config/my_config.json rl_autoschedular_v1
#   sbatch scripts/train/train.sh config/my_config.json --resume results/.../run_0
#   sbatch scripts/train/train.sh config/my_config.json rl_autoschedular_v1 --resume results/.../run_0
#   scripts/submit_and_monitor.sh scripts/train/train.sh config/my_config.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SLURM_SUBMIT_DIR is set by Slurm to the directory where sbatch was called.
# Fall back to deriving from BASH_SOURCE only for local (non-Slurm) runs.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Slurm nodes start with a stripped PATH; restore standard utilities before activating the venv
export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"

source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

# Extract --resume flag early, before consuming positional args
RESUME_FROM=""
POS_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--resume" ]]; then
        _expect_resume=1
    elif [[ -n "${_expect_resume:-}" ]]; then
        RESUME_FROM="$arg"
        _expect_resume=""
    else
        POS_ARGS+=("$arg")
    fi
done
export RESUME_FROM

# Replay positional args so $1/$2/$3 match the original semantics
set -- "${POS_ARGS[@]}"

# Accept config path as first positional argument, default to train1.json.
# Array mode: if SLURM_ARRAY_TASK_ID is set and no explicit config given,
# auto-select from the version list (task 0→v1, 1→v2, 2→v3, 3→v4).
VERSIONS=(baseline v1 v2 v3 v4)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${1:-}" ]]; then
    VERSION="${VERSIONS[$SLURM_ARRAY_TASK_ID]}"
    CONFIG="$PROJECT_ROOT/config/old_dataset/train/${VERSION}.json"
else
    CONFIG="${1:-$PROJECT_ROOT/config/old_dataset/train/train1.json}"
    if [[ "$CONFIG" != /* ]]; then
        CONFIG="$PROJECT_ROOT/$CONFIG"
    fi
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

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Training started at $(date)"
echo "Config: $CONFIG_FILE_PATH"
echo "Implementation: $AUTOSCHEDULER_IMPL"
[[ -n "$RESUME_FROM" ]] && echo "Resume from: $RESUME_FROM"
echo "Node: $(hostname)"
echo "=========================================="

python scripts/train/train.py

echo "Training completed at $(date)"
