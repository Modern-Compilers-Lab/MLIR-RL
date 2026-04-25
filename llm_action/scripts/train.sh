#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J train
#SBATCH -p compute
#SBATCH --reservation=c2
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
export DASK_NODES=${DASK_NODES:-4}

python -m llm_action.src.rl.train_ppo "$@"

# TO EXEC
# sbatch llm_action/scripts/train.sh --max-steps 5 --action-version v10 --benchmarks-name matmuls_12
# sbatch llm_action/scripts/train.sh --max-steps 5 --action-version v10 --param-mode two_policy

# sbatch llm_action/scripts/train.sh --max-steps 8 --action-version v10 --benchmarks-name matmul --ent-coef 0.25 --history-mode include-all --dask-nodes 2
# sbatch llm_action/scripts/train.sh --max-steps 8 --action-version v10 --benchmarks-name matmul --ent-coef 0.25 --history-mode success-encoding --dask-nodes 2
# sbatch llm_action/scripts/train.sh --max-steps 8 --action-version v10 --benchmarks-name matmul --ent-coef 0.1 --history-mode success-encoding --dask-nodes 2
# sbatch llm_action/scripts/train.sh --max-steps 5 --action-version v10 --benchmarks-name matmul --ent-coef 0.25 --history-mode ignore-failed --dask-nodes 2

# sbatch llm_action/scripts/train.sh --max-steps 5 --action-version v10 --benchmarks-name matmul_paper --ent-coef 0.1 --history-mode include-all --dask-nodes 2
# sbatch llm_action/scripts/train.sh --max-steps 5 --action-version v10 --benchmarks-name matmul_paper --history-mode success-encoding --dask-nodes 2 --reward-scale raw --reward-baseline torch

# sbatch llm_action/scripts/train.sh --max-steps 8 --action-version v13 --benchmarks-name matmul_single_paper --ent-coef 0.1 --history-mode success-encoding --dask-nodes 2
# sbatch llm_action/scripts/train.sh --max-steps 8 --action-version v13 --benchmarks-name matmul_single_paper --ent-coef 0.2 --history-mode success-encoding --dask-nodes 2

# TODO!
# sbatch llm_action/scripts/train.sh --max-steps 6 --action-version v13 --benchmarks-name matmul_single_paper --history-mode include-all --dask-nodes 2 --reward-mode intermediate

# Resume from checkpoint:
# sbatch llm_action/scripts/train.sh --resume <log_dir>/checkpoints/ppo_mlir_<N>_steps.zip --max-steps 8 --action-version v13 --benchmarks-name matmul_single_paper --dask-nodes 2