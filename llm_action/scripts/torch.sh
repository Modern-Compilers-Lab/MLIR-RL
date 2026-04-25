#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J torch_exec
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 00:10:00
#SBATCH -o llm_action/logs/jobs/torch/%j.out
#SBATCH -e llm_action/logs/jobs/torch/%j.err
#SBATCH --mail-user=kb5213@nyu.edu
#SBATCH --mail-type=FAIL

# Resource requirement commands end here

# Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
# conda activate llvm-build
conda activate mlir

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
# Usage:
#   sbatch llm_action/scripts/torch.sh matmul <M> <K> <N> [--dtype float64] [--fill-value 0.0] [--warmup-iters 5] [--bench-iters 5]
#   sbatch llm_action/scripts/torch.sh conv2d <N> <C> <H> <W> <F> <KH> <KW> <OH> <OW> [--dtype float64] [--fill-value 0.0] [--warmup-iters 5] [--bench-iters 5]
#
# Examples:
#   sbatch llm_action/scripts/torch.sh matmul 256 256 512
#   sbatch llm_action/scripts/torch.sh matmul 256 512 1024 --dtype float32 --bench-iters 10
#   sbatch llm_action/scripts/torch.sh conv2d 128 32 7 7 256 1 1 7 7
#   sbatch llm_action/scripts/torch.sh conv2d 64 64 56 56 128 3 3 56 56 --dtype float32
python llm_action/src/execution/torch_execution.py "$@"
