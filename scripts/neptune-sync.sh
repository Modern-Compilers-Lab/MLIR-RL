#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 07-00
#SBATCH --mail-type=FAIL,TIME_LIMIT
#SBATCH --mail-user=mt5383@nyu.edu

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate testenv

# Execute the code
python $SCRATCH/MLIR-RL/neptune_sync.py
