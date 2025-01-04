#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 02-00
#SBATCH -o /scratch/na3758/MLIR-RL/scripts/hierarchical.out
#SBATCH -e /scratch/na3758/MLIR-RL/scripts/hierarchical.err

#Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

#Activate any environments if required
conda activate mlir-env

#Execute the code
python /scratch/na3758/MLIR-RL/hierarchical_train.py
