#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J eval_matmul
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH --mem=100G
#SBATCH -t 1-00
#SBATCH -o logs/%x_%j.out

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate main

# Execute the code
# export OMP_NUM_THREADS=12

echo "Base:"
# TIME_BASE=$(mlir-opt matmul.mlir -test-transform-dialect-erase-schedule | python run.py)
TIME_BASE=17707650426
echo "Execution time (ns): $TIME_BASE"

echo "Optimized:"
TIME_OPT=$(mlir-opt n_matmul.mlir -transform-interpreter -test-transform-dialect-erase-schedule | python run.py)
echo "Execution time (ns): $TIME_OPT"

echo "PyTorch:"
conda activate torch-cpu
TIME_TORCH=$(python torch_matmul.py)
echo "Execution time (ns): $TIME_TORCH"
echo "--------------------------"
echo "Speedup over Base: $(echo "scale=4; $TIME_BASE / $TIME_OPT" | bc)x"
echo "Slowdown compared to PyTorch: $(echo "scale=4; $TIME_OPT / $TIME_TORCH" | bc)x"
