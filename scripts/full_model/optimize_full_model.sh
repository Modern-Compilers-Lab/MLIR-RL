#!/bin/bash
#SBATCH --job-name=mlir-fm-opt
#SBATCH --partition=compute
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/fm_opt_%A_%a.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/fm_opt_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Full-model RL optimization — array job.
# Each task processes one neural network model.
#
# Usage:
#   sbatch --array=0-19 scripts/optimize_full_model.sh
#
#   # Override config or checkpoint:
#   sbatch --array=0-19 scripts/optimize_full_model.sh config/my.json /path/to/model.pt
#
#   # Process specific models by name:
#   sbatch scripts/optimize_full_model.sh config/old_dataset/full_model/full_model_optim.json /path/to/model.pt gcn distilbert
#
#   # Submit + monitor:
#   scripts/submit_and_monitor.sh scripts/optimize_full_model.sh

set -e
trap 'echo "FULL-MODEL OPTIMIZATION FAILED"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# ---- environment -----------------------------------------------------------
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

# ---- model list ------------------------------------------------------------
# All 20 neural network models with *_linalg.mlir files in data/nn/raw_bench/
ALL_MODELS=(
    albert
    bart
    bert
    convnext_tiny
    deberta
    densenet121
    distilbert
    efficientnet_b0
    gat
    gcn
    gpt2
    lstm
    mobilenet_v3_small
    resnet18
    resnet50
    resnext50
    roberta
    t5
    vgg11
    vit_b_16
)

# ---- config ----------------------------------------------------------------
CONFIG_FILE="${1:-$PROJECT_ROOT/config/old_dataset/full_model/full_model_optim.json}"
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi
CHECKPOINT="${2:-$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models/model_715.pt}"

# ---- model selection -------------------------------------------------------
# If model names are passed as positional args 3+, use those directly.
# Otherwise, use SLURM_ARRAY_TASK_ID to pick from ALL_MODELS.
if [[ $# -ge 3 ]]; then
    shift 2
    SELECTED_MODELS=("$@")
else
    if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
        SELECTED_MODELS=("${ALL_MODELS[$SLURM_ARRAY_TASK_ID]}")
    else
        echo "ERROR: Set SLURM_ARRAY_TASK_ID or pass model names as arguments."
        exit 1
    fi
fi

export CONFIG_FILE_PATH="$CONFIG_FILE"
export AUTOSCHEDULER_IMPL="rl_autoschedular_v4_5"
export EXEC_TIMEOUT=300
export EXEC_TIMEOUT_CMD=7200
# EXEC_TIMEOUT_CMD is used by the multiprocess fallback in optimize_full_model.py
# when v4_5 bindings time out (300s cap). Large models with dense_resource
# constants need longer for the LLVM pass pipeline to complete.

MODEL_STR=$(IFS=, ; echo "${SELECTED_MODELS[*]}")

echo "=========================================="
echo "Full-Model Optimization started at $(date)"
echo "Config:       $CONFIG_FILE_PATH"
echo "Checkpoint:   $CHECKPOINT"
echo "Models:       $MODEL_STR"
echo "Node:         $(hostname)"
echo "Array task:   ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "=========================================="

cd "$PROJECT_ROOT"

# Output to chunked JSON keyed by array task id
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    OUTPUT="results/full_model/chunk_${SLURM_ARRAY_TASK_ID}.json"
else
    OUTPUT="results/full_model/results_manual.json"
fi

mkdir -p "$(dirname "$OUTPUT")"

python scripts/full_model/optimize_full_model.py \
    --checkpoint "$CHECKPOINT" \
    --models ${SELECTED_MODELS[@]} \
    --output "$OUTPUT" \
    --skip-tagging \
    --resume

echo "Full-Model Optimization completed at $(date)"
echo "Results → $OUTPUT"

# ---- merge hint ------------------------------------------------------------
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    MERGE_SCRIPT="$PROJECT_ROOT/scripts/merge_full_model_results.sh"
    echo ""
    echo "To merge all chunks into a single results file, run after all array tasks finish:"
    echo "  bash $MERGE_SCRIPT"
fi
