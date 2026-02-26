#!/bin/bash

# Define the resource requirements here using #SBATCH
# Mirrors torch.sh exactly for fair benchmarking.

#SBATCH -J mlir_exec
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --qos=c2
#SBATCH --exclusive
#SBATCH -c 16
#SBATCH --mem=100G
#SBATCH -t 00:10:00
#SBATCH -o llm_action/logs/jobs/mlir_%j.out
#SBATCH -e llm_action/logs/jobs/mlir_%j.err

# Resource requirement commands end here

# Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate llvm-build

# Source MLIR-specific environment (PYTHONPATH, MLIR_SHARED_LIBS, etc.)
source scripts/setup_env.sh

# Thread-affinity settings (critical for SLURM performance)
# Without these, OpenMP/MKL threads migrate across NUMA domains
# causing massive cache-thrashing slowdowns.
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Execute the code
# Usage: sbatch llm_action/scripts/mlir.sh <code_file> [--transform-file <path>] [--pass-pipeline <pipeline>]
python llm_action/src/execution/mlir_execution.py "$@"
