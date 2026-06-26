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
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260529_191729_v48_dataset_matmul_free
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_032102_v48_dataset_matmul_free_interm
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260529_191812_v48_dataset_matmul_graph
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_032115_v48_dataset_matmul_graph_interm

# Conv2d
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260520_193712_v33_dataset_conv2d
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260529_191949_v49_dataset_conv2d_free
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_050304_v49_dataset_conv2d_free_interm
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_042634_v49_dataset_conv2d_graph
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_050311_v49_dataset_conv2d_graph_interm

# Pooling
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260520_201653_v34_dataset_pooling
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260531_012742_v50_dataset_pooling_free
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260531_012748_v50_dataset_pooling_graph

# Add
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260522_201059_v35_dataset_add
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_145351_v51_dataset_add_free
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_145357_v51_dataset_add_graph

# ReLu
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260522_210639_v37_dataset_relu
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_145246_v52_dataset_relu_free
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260530_145252_v52_dataset_relu_graph
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260603_024434_v52_dataset_relu_free
## sbatch llm_action/scripts/evaluate.sh --run-name ppo_20260603_024449_v52_dataset_relu_graph
