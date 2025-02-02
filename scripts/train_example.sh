#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 07-00
#SBATCH -o /scratch/$NYU_NET_ID/MLIR-RL/logs/train.out
#SBATCH -e /scratch/$NYU_NET_ID/MLIR-RL/logs/train.err

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate $CONDA_ENV_NAME

# Execute the code
python /scratch/$NYU_NET_ID/MLIR-RL/train.py
