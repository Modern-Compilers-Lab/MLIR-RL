#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 07-00
#SBATCH -o output/slurm-%j.out

# Load miniconda
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate $CONDA_ENV_NAME

# Execute the code
python upload_logs.py --dir_path $1 --algorithm $2
