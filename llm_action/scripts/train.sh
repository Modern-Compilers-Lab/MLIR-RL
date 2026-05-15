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

# Dask workers: set DASK_NODES to use persistent compute workers instead of per-step sbatch jobs. Requires --executor-type dask.
export DASK_NODES=${DASK_NODES:-2}

python -m llm_action.src.rl.train_ppo "$@"

# Usage

# Matmul Dataset
## sbatch llm_action/scripts/train.sh --action-version v29 --benchmarks-name dataset_matmul --exp-name v29_dataset_matmul_ent_decay

# Conv2d Dataset
## sbatch llm_action/scripts/train.sh --action-version v30 --benchmarks-name dataset_conv2d --exp-name v30_dataset_conv2d --ent-coef 0.0025 --ent-coef-final 0.0001 --action-head-init zero

# Pooling Dataset
## sbatch llm_action/scripts/train.sh --action-version v31 --benchmarks-name dataset_pooling --exp-name v31_dataset_pooling --ent-coef 0.01 --ent-coef-final 0.0001 --max-steps 5