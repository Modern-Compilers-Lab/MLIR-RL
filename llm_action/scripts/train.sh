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
export DASK_NODES=${DASK_NODES:-1}

python -m llm_action.src.rl.train_ppo "$@"

# Usage

# Matmul Dataset
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_free --masking-mode none --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul__ent01_free --masking-mode none --max-steps 4 --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_graph --masking-mode schedule_graph --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul__ent01_graph --masking-mode schedule_graph --max-steps 4 --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_free_interm --masking-mode none --max-steps 4 --reward-mode intermediate
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_graph_interm --masking-mode schedule_graph --max-steps 4 --reward-mode intermediate

# Conv2d Dataset
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_free --masking-mode none --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_free_interm --masking-mode none --max-steps 5 --reward-mode intermediate
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_graph --masking-mode schedule_graph --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_graph_256x3 --masking-mode schedule_graph --max-steps 5 --net-arch 256 256 256
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_graph_interm --masking-mode schedule_graph --max-steps 5 --reward-mode intermediate

# Pooling Dataset
## sbatch llm_action/scripts/train.sh --action-version v50 --benchmarks-name dataset_pooling --exp-name v50_dataset_pooling_free --masking-mode none --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v50 --benchmarks-name dataset_pooling --exp-name v50_dataset_pooling_graph --masking-mode schedule_graph --max-steps 5

# Add Dataset
## sbatch llm_action/scripts/train.sh --action-version v51 --benchmarks-name dataset_add --exp-name v51_dataset_add_free --masking-mode none --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v51 --benchmarks-name dataset_add --exp-name v51_dataset_add_graph --masking-mode schedule_graph --max-steps 4

# ReLu Dataset
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_free --masking-mode none --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_free_ent01 --masking-mode none --max-steps 4 --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_free_ent0025 --masking-mode none --max-steps 4 --ent-coef 0.0025 --ent-coef-final 0.000025
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_graph --masking-mode schedule_graph --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_graph_ent01 --masking-mode schedule_graph --max-steps 4 --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_graph_ent0025 --masking-mode schedule_graph --max-steps 4 --ent-coef 0.0025 --ent-coef-final 0.000025