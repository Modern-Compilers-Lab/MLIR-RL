#!/bin/bash

#Define the resource requirements here using #SBATCH

#SBATCH -p compute
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH -c 28
#SBATCH --mem=64G
#SBATCH -t 02-00
#SBATCH -o /scratch/mt5383/MLIR-RL/scripts/hierarchical.out
#SBATCH -e /scratch/mt5383/MLIR-RL/scripts/hierarchical.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mt5383@nyu.edu

#Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc
eval "$(conda shell.bash hook)"

#Activate any environments if required
conda activate main

#Execute the code
python /scratch/mt5383/MLIR-RL/hierarchical_train.py
