#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 07-00
#SBATCH -o /scratch/mt5383/MLIR-RL/logs/neptune-sync.out
#SBATCH -e /scratch/mt5383/MLIR-RL/logs/neptune-sync.err
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=mt5383@nyu.edu

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate testenv

# Execute the code
export NEPTUNE_PROJECT=PFE-NYUAD/mlir-rl
python $SCRATCH/MLIR-RL/neptune_sync.py
