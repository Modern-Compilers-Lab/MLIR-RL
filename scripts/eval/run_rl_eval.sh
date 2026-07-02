#!/bin/bash
#SBATCH --job-name=rl-eval
#SBATCH --partition=compute
#SBATCH -C dalma
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-06:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/rl_eval_%A_%a.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/rl_eval_%A_%a.err
#SBATCH --mail-type=END,FAIL
#
# Usage:
#   sbatch --array=0-17 scripts/eval/run_rl_eval.sh 700
#   sbatch --array=0-17 scripts/eval/run_rl_eval.sh 800
#   ...
#   sbatch --array=0-17 scripts/eval/run_rl_eval.sh 1999
#
# Single-checkpoint RL evaluation for 18 models.
# Cluster: jubail (AMD Zen 2, 128 cores, 480 GB)
# Output: results/full_model_1/rl_opt_<CKPT>/
# Resume via markers: results/full_model_1/rl_opt_<CKPT>_markers/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

CKPT_NUM="${1:?Usage: sbatch --array=0-17 scripts/eval/run_rl_eval.sh <checkpoint_number>}"
MODEL_PT="model_${CKPT_NUM}.pt"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

MODELS=(
    "albert"    "bart"      "bert"          "convnext_tiny" "densenet121"
    "distilbert" "efficientnet_b0" "gat"   "gcn"           "gpt2-large"
    "lstm"      "mobilenet_v3_small" "resnet18" "resnet50" "resnext50"
    "t5"        "vgg11"     "vit_b_16"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
MODEL="${MODELS[$TASK_ID]}"

CKPT="$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models/${MODEL_PT}"
BASELINES="$PROJECT_ROOT/results/full_model_dalma/all_blocks_baselines.json"
OUTPUT_DIR="$PROJECT_ROOT/results/full_model_dalma/rl_opt_${CKPT_NUM}"
OUTPUT="$OUTPUT_DIR/rl_opt_${CKPT_NUM}_${MODEL}.json"
MARKERS="$PROJECT_ROOT/results/full_model_dalma/rl_opt_${CKPT_NUM}_markers"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" "$MARKERS"

echo "=========================================="
echo "RL Evaluation (ckpt $CKPT_NUM) — $(date)"
echo "Cluster:  jubail"
echo "Task:     $TASK_ID / ${#MODELS[@]}"
echo "Model:    $MODEL"
echo "Checkpoint: $CKPT"
echo "Output:   $OUTPUT"
echo "Markers:  $MARKERS"
echo "Node:     $(hostname)"
echo "=========================================="

export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5
export CONFIG_FILE_PATH="$PROJECT_ROOT/config/old_dataset/train/baseline.json"

python scripts/full_model/optimize_model_via_blocks.py \
    --model "$MODEL" \
    --single-checkpoint "$CKPT" \
    --baselines "$BASELINES" \
    --output "$OUTPUT" \
    --markers-dir "$MARKERS" \
    --blocks-dir data/nn/blocks \
    --skip-extraction

echo ""
echo "Completed at $(date)"
