#!/bin/bash
#SBATCH --job-name=mlir-eval
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=28
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/eval_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch scripts/eval.sh                                                    # array mode: auto-picks version
#   sbatch scripts/eval.sh config/old_dataset/train/baseline.json             # single version from config
#   sbatch scripts/eval.sh config/old_dataset/train/baseline.json v1          # explicit version
#   sbatch scripts/eval.sh config/new_dataset/eval/v4_6_eval.json --checkpoint 100  # single checkpoint eval

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

export MIN_EXEC_TIMEOUT=${MIN_EXEC_TIMEOUT:-300}

# Extract --checkpoint flag before consuming positional args
EVAL_CHECKPOINT=""
POS_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--checkpoint" ]]; then
        _expect_checkpoint=1
    elif [[ -n "${_expect_checkpoint:-}" ]]; then
        EVAL_CHECKPOINT="$arg"
        _expect_checkpoint=""
    else
        POS_ARGS+=("$arg")
    fi
done
export EVAL_CHECKPOINT

# Replay positional args
set -- "${POS_ARGS[@]}"

# Version resolution: array task > positional args > config file
ALL_VERSIONS=(v0 v1 v2 v3 v4)
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    VERSIONS=("${ALL_VERSIONS[$SLURM_ARRAY_TASK_ID]}")
elif [[ $# -ge 2 ]]; then
    CONFIG_FILE="$1"
    shift
    VERSIONS=("$@")
else
    CONFIG_FILE="${1:-$PROJECT_ROOT/config/old_dataset/train/baseline.json}"
    VERSIONS=()
fi

CONFIG="${CONFIG_FILE:-$PROJECT_ROOT/config/old_dataset/train/baseline.json}"
if [[ "$CONFIG" != /* ]]; then
    CONFIG="$PROJECT_ROOT/$CONFIG"
fi

if [[ ${#VERSIONS[@]} -eq 0 ]]; then
    CONFIG_IMPL=$(python3 - <<PY
import json
try:
    with open("$CONFIG", "r") as f:
        print((json.load(f).get("implementation") or "").strip())
except Exception:
    print("")
PY
    )
    IMPL="${CONFIG_IMPL:-rl_autoschedular_v0}"
    VERSIONS=("${IMPL#rl_autoschedular_}")
fi

for VERSION in "${VERSIONS[@]}"; do
    IMPLEMENTATION="rl_autoschedular_${VERSION}"
    export AUTOSCHEDULER_IMPL="$IMPLEMENTATION"
    export CONFIG_FILE_PATH="$CONFIG"

    # Optional run ID for logging (resumption of run_0)
    if [[ -n "${FORCE_RUN_ID:-}" ]]; then
        export FORCE_RUN_ID="$FORCE_RUN_ID"
    fi

    # If --checkpoint mode: EVAL_DIR = <results_dir>/models/
    if [[ -n "${EVAL_CHECKPOINT:-}" ]]; then
        RESULTS_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['results_dir'])")
        MODELS_DIR="$PROJECT_ROOT/$RESULTS_DIR/models"
        if [[ -d "$MODELS_DIR" ]]; then
            export EVAL_DIR="$MODELS_DIR"
            export EVAL_START="$EVAL_CHECKPOINT"
            export EVAL_END="$EVAL_CHECKPOINT"
            export EVAL_STRIDE=1
            export FORCE_RUN_ID="ckpt_${EVAL_CHECKPOINT}"
        else
            echo "ERROR: models/ directory not found at $MODELS_DIR"
            exit 1
        fi
    fi

    echo "=========================================="
    echo "Evaluation started at $(date)"
    echo "Version:  $VERSION ($IMPLEMENTATION)"
    echo "Config:   $CONFIG_FILE_PATH"
    [[ -n "${EVAL_CHECKPOINT:-}" ]] && echo "Checkpoint: $EVAL_CHECKPOINT"
    echo "Node:     $(hostname)"
    echo "=========================================="

    # Derive EVAL_DIR: ONLY if not already set by environment
    if [[ -z "${EVAL_DIR:-}" ]]; then
        RESULTS_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['results_dir'])")
        export EVAL_DIR="$PROJECT_ROOT/$RESULTS_DIR/models"
    fi
    echo "EVAL_DIR: $EVAL_DIR"

    cd "$PROJECT_ROOT"
    python scripts/eval/eval.py

    echo "Evaluation completed at $(date)"
    echo ""
done
