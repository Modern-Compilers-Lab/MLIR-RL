#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J train
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --qos=c2
#SBATCH -c 28
#SBATCH --mem=100G
#SBATCH -t 05-00
#SBATCH -o llm_action/logs/jobs/train/%j.out
#SBATCH -e llm_action/logs/jobs/train/%j.err
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

# Dask workers: set DASK_NODES to use persistent compute workers instead of
# per-step sbatch jobs. Requires --executor-type dask.
export DASK_NODES=${DASK_NODES:-2}

# Usage:
#   sbatch llm_action/scripts/train.sh                                           # slurm executor (legacy)
#   sbatch llm_action/scripts/train.sh --executor-type dask                      # dask executor (recommended)
#   sbatch llm_action/scripts/train.sh --executor-type dask --total-timesteps 500  # quick test
python -m llm_action.src.rl.train_ppo "$@"

# python -m llm_action.src.rl.train_ppo --max-steps 6 --action-version v6 --executor-type local
# sbatch llm_action/scripts/train.sh --max-steps 6 --action-version v6 --executor-type dask
# 
# TO EXEC IN THE MORNING
# sbatch llm_action/scripts/train.sh --max-steps 5 --action-version v7 --executor-type dask --reward-scale log
# sbatch llm_action/scripts/train.sh --max-steps 7 --action-version v10 --executor-type dask --reward-scale log


# python -m llm_action.src.rl.train_ppo --max-steps 7 --action-version v8 --executor-type local

# sbatch llm_action/scripts/train.sh --max-steps 7 --action-version v10 --executor-type dask --reward-scale log --reward-mode intermediate