#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J evaluate
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --qos=c2
#SBATCH -c 28
#SBATCH --mem=100G
#SBATCH -t 01-00
#SBATCH -o llm_action/logs/jobs/evaluate/%j.out
#SBATCH -e llm_action/logs/jobs/evaluate/%j.err
#SBATCH --mail-user=kb5213@nyu.edu
#SBATCH --mail-type=ALL

# Resource requirement commands end here

# Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"
conda activate mlir

# Thread-affinity for any local MLIR execution
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=0

# Dask workers: one server is sufficient for evaluation (evaluate_ppo defaults --dask-nodes 1).
export DASK_NODES=${DASK_NODES:-1}

python -m llm_action.src.rl.evaluate_ppo "$@"

# Matmul
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260520_170930_v32_dataset_matmul --mode training-logs

# Conv2d
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260520_193712_v33_dataset_conv2d --mode training-logs

# Pooling
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260520_201653_v34_dataset_pooling --mode training-logs

# Add
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260522_201059_v35_dataset_add --mode training-logs

# ReLu
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260522_210639_v37_dataset_relu --mode training-logs
