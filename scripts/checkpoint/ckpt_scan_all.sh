#!/bin/bash
#SBATCH --job-name=mlir-ckpt-scan
#SBATCH --partition=compute
#SBATCH --mem=300G
#SBATCH --cpus-per-task=128
#SBATCH -C bergamo
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/ckpt_scan_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/ckpt_scan_%j.err
#SBATCH --mail-type=END,FAIL
#
# Checkpoint scan: evaluates all 19 models across 14 checkpoints (700-1999 step 100)
# using block-based approach with parallel block evaluation.
#
# All 19 models run in parallel, each getting ~6 CPUs.
# Results saved to results/full_model/scan/<model>_scan.json
#
# Usage:
#   sbatch scripts/ckpt_scan_all.sh
#
#   # Override config or checkpoint range:
#   sbatch scripts/ckpt_scan_all.sh config/old_dataset/full_model/full_model_optim.json
#
#   # Monitor:
#   tail -f logs/ckpt_scan_<jobid>.out

set -e
trap 'echo "CHECKPOINT SCAN FAILED"' ERR

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

# ---- config ----------------------------------------------------------------
CONFIG_FILE="${1:-$PROJECT_ROOT/config/old_dataset/full_model/full_model_optim.json}"
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$PROJECT_ROOT/$CONFIG_FILE"
fi

export CONFIG_FILE_PATH="$CONFIG_FILE"
export AUTOSCHEDULER_IMPL="rl_autoschedular_v4_5"

# ---- checkpoints -----------------------------------------------------------
MODELS_DIR="$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models"
CHECKPOINTS=""
for i in $(seq 700 100 1900) 1999; do
    ckpt="$MODELS_DIR/model_${i}.pt"
    if [[ -f "$ckpt" ]]; then
        CHECKPOINTS="$CHECKPOINTS $ckpt"
    fi
done

if [[ -z "$CHECKPOINTS" ]]; then
    echo "ERROR: No checkpoints found in $MODELS_DIR"
    exit 1
fi

echo "Found checkpoints: $CHECKPOINTS"

# ---- models ----------------------------------------------------------------
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
    lstm
    mobilenet_v3_small
    resnet18
    resnet50
    resnext50
    t5
    vgg11
    vit_b_16
)

# ---- parallelism -----------------------------------------------------------
NUM_MODELS=${#ALL_MODELS[@]}
NUM_CPUS=${SLURM_CPUS_PER_TASK:-128}
# Use sequential (workers=1) — parallel workers fail silently with MLIR bindings
WORKERS_PER_MODEL=1

SCAN_DIR="$PROJECT_ROOT/results/full_model/scan"
mkdir -p "$SCAN_DIR"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Checkpoint Scan started at $(date)"
echo "Config:       $CONFIG_FILE_PATH"
echo "Models:       $NUM_MODELS"
echo "CPUs:         $NUM_CPUS"
echo "Workers/model: $WORKERS_PER_MODEL"
echo "Node:         $(hostname)"
echo "=========================================="

# ---- launch all models in parallel -----------------------------------------
PIDS=()
for model in "${ALL_MODELS[@]}"; do
    OUTPUT="$SCAN_DIR/${model}_scan.json"
    LOG="$SCAN_DIR/${model}.log"

    echo "[$(date)] Launching $model (workers=$WORKERS_PER_MODEL) → $OUTPUT"

    python scripts/full_model/optimize_model_via_blocks.py \
        --checkpoints $CHECKPOINTS \
        --model "$model" \
        --models-dir "$PROJECT_ROOT/data/nn/raw_bench" \
        --blocks-dir "$PROJECT_ROOT/data/nn/blocks" \
        --output "$OUTPUT" \
        --window-size 5 \
        --stride 3 \
        --skip-extraction \
        --workers $WORKERS_PER_MODEL \
        --baselines-json "$PROJECT_ROOT/results/full_model/baselines/blocks_baseline.json" \
        > "$LOG" 2>&1 &

    PIDS+=($!)
done

# ---- wait for all ----------------------------------------------------------
FAILED=0
for i in "${!PIDS[@]}"; do
    model="${ALL_MODELS[$i]}"
    pid="${PIDS[$i]}"
    if wait $pid; then
        echo "[$(date)] ✓ $model completed"
    else
        echo "[$(date)] ✗ $model FAILED (pid=$pid)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "Checkpoint Scan completed at $(date)"
echo "Failed: $FAILED / $NUM_MODELS"
echo "Results → $SCAN_DIR/"
echo "=========================================="

# ---- merge -----------------------------------------------------------------
echo ""
echo "Merging results..."
python scripts/checkpoint/merge_ckpt_scan.py --scan-dir "$SCAN_DIR" --output "$SCAN_DIR/summary.json"

if [[ $FAILED -gt 0 ]]; then
    exit 1
fi
