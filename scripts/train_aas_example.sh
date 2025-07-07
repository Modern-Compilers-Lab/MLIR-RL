#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 07-00
#SBATCH -o logs/train.out
#SBATCH -e logs/train.err

# Load miniconda
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate $CONDA_ENV_NAME

# Set config file path
export AAS_CONFIG_FILE_PATH=config/aas_example.json
export OMP_NUM_THREADS=8
# Set save file path for agent's network
export AAS_SAVE_FILE_PATH=models/aas_agent.pt
# Execute the code
python train_aas.py
