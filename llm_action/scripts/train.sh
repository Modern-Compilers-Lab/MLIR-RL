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
## sbatch llm_action/scripts/train.sh --action-version v32 --benchmarks-name dataset_matmul --exp-name v32_dataset_matmul
## sbatch llm_action/scripts/train.sh --action-version v32 --benchmarks-name dataset_matmul --exp-name v32_dataset_matmul_ent01 --ent-coef 0.01 --ent-coef-final 0.0

# Conv2d Dataset
## sbatch llm_action/scripts/train.sh --action-version v33 --benchmarks-name dataset_conv2d --exp-name v33_dataset_conv2d_ent0025 --ent-coef 0.0025 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v36 --benchmarks-name dataset_conv2d_img2col --exp-name v36_dataset_conv2d_img2col_ent01 --ent-coef 0.01 --ent-coef-final 0.0001

# Pooling Dataset
## sbatch llm_action/scripts/train.sh --action-version v34 --benchmarks-name dataset_pooling --exp-name v34_dataset_pooling --ent-coef 0.005 --ent-coef-final 0.0

# Add Dataset
## sbatch llm_action/scripts/train.sh --action-version v35 --benchmarks-name dataset_add --exp-name v35_dataset_add --ent-coef 0.01 --ent-coef-final 0.0001

# ReLu Dataset
## sbatch llm_action/scripts/train.sh --action-version v37 --benchmarks-name dataset_relu --exp-name v37_dataset_relu --ent-coef 0.01 --ent-coef-final 0.0001