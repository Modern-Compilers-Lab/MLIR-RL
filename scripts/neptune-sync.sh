#!/bin/bash

# Define the resource requirements here using #SBATCH

# SBATCH -j neptune_sync
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 07-00
#SBATCH -o logs/neptune/%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kb5213@nyu.edu

# Resource requirements end here

# Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate llvm-build

# Execute the code
python $SCRATCH/MLIR-RL/neptune_sync.py
