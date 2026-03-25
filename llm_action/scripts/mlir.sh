#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J mlir_exec
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o llm_action/logs/jobs/mlir/%j.out
#SBATCH -e llm_action/logs/jobs/mlir/%j.err
#SBATCH --mail-user=kb5213@nyu.edu
#SBATCH --mail-type=FAIL

# Resource requirement commands end here

# Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
# conda activate llvm-build
conda activate mlir

# Source MLIR-specific environment (PYTHONPATH, MLIR_SHARED_LIBS, etc.)
# source scripts/setup_env.sh

# Thread-affinity settings (critical for SLURM performance)
# Without these, OpenMP/MKL threads migrate across NUMA domains
# causing massive cache-thrashing slowdowns.
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0

# Execute the code
# Usage: sbatch llm_action/scripts/mlir.sh <code_file> [--transform-file <path>] [--pass-pipeline <pipeline>]
python llm_action/src/execution/mlir_execution.py "$@"
