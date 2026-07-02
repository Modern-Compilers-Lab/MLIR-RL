#!/bin/bash
#SBATCH --job-name=pt-times
#SBATCH --partition=compute
#SBATCH -C jubail
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=0-04:00:00
#SBATCH --output=/scratch/mb10856/MLIR-RL/logs/pt_times_%j.out
#SBATCH --error=/scratch/mb10856/MLIR-RL/logs/pt_times_%j.err
#SBATCH --mail-type=END,FAIL
#
# PyTorch eager + JIT timing for all 22 models on Jubail.
# Output: results/full_model_1/pytorch_times.json

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export PATH="/usr/local/bin:/usr/bin:/bin:$PATH"
source "${CONDA_ENV:-$HOME/envs/mlir/bin/activate}"
export LD_LIBRARY_PATH=$HOME/envs/mlir/lib:$LD_LIBRARY_PATH
export PYTHONPATH="$LLVM_BUILD_PATH/tools/mlir/python_packages/mlir_core:$PROJECT_ROOT:$PROJECT_ROOT/rl_autoschedular${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJECT_ROOT"

echo "=========================================="
echo "PyTorch Timing — $(date)"
echo "Cluster:  jubail"
echo "Output:   results/full_model_1/pytorch_times.json"
echo "Node:     $(hostname)"
echo "=========================================="

python scripts/baseline/get_pytorch_baselines.py \
    --config results/full_model_1/pytorch_models.json \
    --output results/full_model_1/pytorch_times.json \
    --csv-output results/full_model_1/full_baselines.csv \
    --mlir-baselines results/full_model_1/baselines/full_model.json

echo ""
echo "Completed at $(date)"
