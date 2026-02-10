#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J claude_matmul
#SBATCH -p compute
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 7-00
#SBATCH -o claude_optimization_%j.log

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate main

# Execute the code
for ((i = 0 ; i < 100 ; i++ )); do
    claude --continue --print --verbose --output-format=text "$(cat prompt.txt)";
done
