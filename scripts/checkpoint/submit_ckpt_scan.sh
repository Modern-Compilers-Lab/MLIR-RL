#!/bin/bash
# Submit one Slurm job per model for checkpoint scanning.
# Each job gets 36 CPUs and runs sequentially (workers=1).
#
# Usage:
#   bash scripts/submit_ckpt_scan.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$PROJECT_ROOT"

MODELS=(
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

MODELS_DIR="$PROJECT_ROOT/results/experiment3/v4_5_agent/run_0/models"
CHECKPOINTS=""
for i in $(seq 700 100 1900) 1999; do
    ckpt="$MODELS_DIR/model_${i}.pt"
    if [[ -f "$ckpt" ]]; then
        CHECKPOINTS="$CHECKPOINTS $ckpt"
    fi
done

SCAN_DIR="$PROJECT_ROOT/results/full_model/scan"
mkdir -p "$SCAN_DIR"

echo "Submitting 18 jobs, one per model..."
echo "Checkpoints: $(echo $CHECKPOINTS | wc -w)"
echo ""

for model in "${MODELS[@]}"; do
    OUTPUT="$SCAN_DIR/${model}_scan.json"
    LOG="$SCAN_DIR/${model}.log"

    JOB_ID=$(sbatch --job-name="ckpt-${model}" \
        --partition=compute \
        --mem=16G \
        --cpus-per-task=8 \
        -C bergamo \
        --time=1-00:00:00 \
        --output="$LOG" \
        --error="$LOG.err" \
        --mail-type=END,FAIL \
        --wrap="
            export PYTHONUNBUFFERED=1
            source ~/envs/mlir/bin/activate
            set -a && source $PROJECT_ROOT/.env && set +a
            export PATH=/usr/local/bin:/usr/bin:/bin:\$PATH
            export PYTHONPATH=\$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular\${PYTHONPATH:+:\$PYTHONPATH}
            cd $PROJECT_ROOT
            export CONFIG_FILE_PATH=$PROJECT_ROOT/config/old_dataset/full_model/full_model_optim.json
            export AUTOSCHEDULER_IMPL=rl_autoschedular_v4_5
            export SKIP_MLIR_BINDINGS=1
            python scripts/full_model/optimize_model_via_blocks.py \
                --checkpoints $CHECKPOINTS \
                --model $model \
                --models-dir $PROJECT_ROOT/data/nn/raw_bench \
                --blocks-dir $PROJECT_ROOT/data/nn/blocks \
                --output $OUTPUT \
                --window-size 5 \
                --stride 3 \
                --skip-extraction \
                --workers 1 \
                --baselines-json $PROJECT_ROOT/results/full_model/baselines/blocks_baseline.json
        " | awk '{print $4}')

    echo "  $model → job $JOB_ID"
done

echo ""
echo "All jobs submitted. Monitor with: squeue -u \$USER"
echo "Check individual logs in: $SCAN_DIR/*.log"
