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
## Ablation on sample efficiency
## sbatch llm_action/scripts/train.sh --action-version v42 --benchmarks-name dataset_matmul --exp-name v42_dataset_matmul_free --masking-mode none --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v42 --benchmarks-name dataset_matmul --exp-name v42_dataset_matmul_dep --masking-mode dependencies --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v42 --benchmarks-name dataset_matmul --exp-name v42_dataset_matmul_graph --masking-mode schedule_graph --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v42 --benchmarks-name dataset_matmul --exp-name v42_dataset_matmul_graph_interm --masking-mode schedule_graph --max-steps 4 --reward-mode intermediate

## sbatch llm_action/scripts/train.sh --action-version v32 --benchmarks-name dataset_matmul --exp-name v32_dataset_matmul
## sbatch llm_action/scripts/train.sh --action-version v32 --benchmarks-name dataset_matmul --exp-name v32_dataset_matmul_ent01 --ent-coef 0.01 --ent-coef-final 0.0
## sbatch llm_action/scripts/train.sh --action-version v38 --benchmarks-name dataset_matmul --exp-name v38_dataset_matmul_ent001 --max-steps 5 --ent-coef 0.001 --ent-coef-final 0.00001
## sbatch llm_action/scripts/train.sh --action-version v38 --benchmarks-name dataset_matmul --exp-name v38_dataset_matmul_ent005_delta_vf --max-steps 5 --ent-coef 0.005 --ent-coef-final 0.001 --reward-scale delta --vf-coef 0.01
## sbatch llm_action/scripts/train.sh --action-version v40 --benchmarks-name dataset_matmul --exp-name v40_dataset_matmul_ent005actual_4a --max-steps 4 --ent-coef 0.005 --ent-coef-final 0.0001

## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf05_ent01_final_xl --masking-mode schedule_graph --max-steps 4 --vf-coef 0.05 --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf05_ent005_final_unified --masking-mode schedule_graph --max-steps 4 --vf-coef 0.05 --ent-coef 0.005 --ent-coef-final 0.00005
## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf05_ent0025_final_xl --masking-mode schedule_graph --max-steps 4 --vf-coef 0.05 --ent-coef 0.0025 --ent-coef-final 0.000025

## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf05_ent005_interm_unified --masking-mode schedule_graph --max-steps 4 --vf-coef 0.05 --ent-coef 0.005 --ent-coef-final 0.00005 --reward-mode intermediate
## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf05_ent005_final_unified --masking-mode schedule_graph --max-steps 4 --vf-coef 0.05 --ent-coef 0.005 --ent-coef-final 0.00005 --reward-mode final

## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf05_vec_ent001_intermediate --masking-mode schedule_graph --max-steps 4 --vf-coef 0.05 --ent-coef 0.001 --ent-coef-final 0.00001 --reward-mode intermediate
## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_dep_vf05_vec_ent001_intermediate --masking-mode dependencies --max-steps 4 --vf-coef 0.05 --ent-coef 0.001 --ent-coef-final 0.00001 --reward-mode intermediate

## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_dep_ent005 --masking-mode dependencies --max-steps 4 --ent-coef 0.005 --ent-coef-final 0.00005
## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf05_ep4 --masking-mode schedule_graph --max-steps 4 --vf-coef 0.05
## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_vf5_ep4_intermediate --masking-mode schedule_graph --max-steps 4 --vf-coef 0.5 --n-epochs 4 --reward-mode intermediate
## sbatch llm_action/scripts/train.sh --action-version v41 --benchmarks-name dataset_matmul --exp-name v41_dataset_matmul_graph_ent005 --masking-mode schedule_graph --max-steps 4 --ent-coef 0.005 --ent-coef-final 0.00005

# Shape-conditioning knobs (matmul): per-shape reward + max-normalized bounds + longer entropy decay
## sbatch llm_action/scripts/train.sh --action-version v38 --benchmarks-name dataset_matmul --exp-name v38_matmul_relative --max-steps 5 --reward-scale relative --ent-coef 0.005 --ent-coef-final 0.001
## sbatch llm_action/scripts/train.sh --action-version v38 --benchmarks-name dataset_matmul --exp-name v38_matmul_relative_maxbounds --max-steps 5 --reward-scale relative --loop-bound-encoding max --ent-coef 0.005 --ent-coef-final 0.001
## sbatch llm_action/scripts/train.sh --action-version v38 --benchmarks-name dataset_matmul --exp-name v38_matmul_relative_max_entlong --max-steps 5 --reward-scale relative --loop-bound-encoding max --ent-coef 0.005 --ent-coef-final 0.001 --ent-coef-decay-frac 2.0


-----

## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_free --masking-mode none --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul__ent01_free --masking-mode none --max-steps 4 --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_graph --masking-mode schedule_graph --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul__ent01_graph --masking-mode schedule_graph --max-steps 4 --ent-coef 0.01 --ent-coef-final 0.0001

## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_free_interm --masking-mode none --max-steps 4 --reward-mode intermediate
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_dep --masking-mode dependencies --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v48 --benchmarks-name dataset_matmul --exp-name v48_dataset_matmul_graph_interm --masking-mode schedule_graph --max-steps 4 --reward-mode intermediate

# Conv2d Dataset
## sbatch llm_action/scripts/train.sh --action-version v33 --benchmarks-name dataset_conv2d --exp-name v33_dataset_conv2d_ent0025 --ent-coef 0.0025 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v36 --benchmarks-name dataset_conv2d_img2col --exp-name v36_dataset_conv2d_img2col_ent01 --ent-coef 0.01 --ent-coef-final 0.0001

## sbatch llm_action/scripts/train.sh --action-version v43 --benchmarks-name dataset_conv2d --exp-name v43_dataset_conv2d_free --masking-mode none --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v43 --benchmarks-name dataset_conv2d --exp-name v43_dataset_conv2d_dep --masking-mode dependencies --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v43 --benchmarks-name dataset_conv2d --exp-name v43_dataset_conv2d_graph --masking-mode schedule_graph --max-steps 5

## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_free --masking-mode none --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_free_interm --masking-mode none --max-steps 5 --reward-mode intermediate
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_graph --masking-mode schedule_graph --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_graph_256x2 --masking-mode schedule_graph --max-steps 5 --net-arch 256 256
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_graph_256x3 --masking-mode schedule_graph --max-steps 5 --net-arch 256 256 256
## sbatch llm_action/scripts/train.sh --action-version v49 --benchmarks-name dataset_conv2d --exp-name v49_dataset_conv2d_graph_interm --masking-mode schedule_graph --max-steps 5 --reward-mode intermediate

# Pooling Dataset
## sbatch llm_action/scripts/train.sh --action-version v34 --benchmarks-name dataset_pooling --exp-name v34_dataset_pooling --ent-coef 0.005 --ent-coef-final 0.0
## sbatch llm_action/scripts/train.sh --action-version v50 --benchmarks-name dataset_pooling --exp-name v50_dataset_pooling_free --masking-mode none --max-steps 5
## sbatch llm_action/scripts/train.sh --action-version v50 --benchmarks-name dataset_pooling --exp-name v50_dataset_pooling_graph --masking-mode schedule_graph --max-steps 5


# Add Dataset
## sbatch llm_action/scripts/train.sh --action-version v35 --benchmarks-name dataset_add --exp-name v35_dataset_add --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v51 --benchmarks-name dataset_add --exp-name v51_dataset_add_free --masking-mode none --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v51 --benchmarks-name dataset_add --exp-name v51_dataset_add_graph --masking-mode schedule_graph --max-steps 4


# ReLu Dataset
## sbatch llm_action/scripts/train.sh --action-version v37 --benchmarks-name dataset_relu --exp-name v37_dataset_relu --ent-coef 0.01 --ent-coef-final 0.0001
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_free --masking-mode none --max-steps 4
## sbatch llm_action/scripts/train.sh --action-version v52 --benchmarks-name dataset_relu --exp-name v52_dataset_relu_graph --masking-mode schedule_graph --max-steps 4

# ML Dataset
## sbatch llm_action/scripts/train.sh --action-version v45 --benchmarks-name dataset_ml --exp-name v45_dataset_ml_free --max-steps 5 --masking-mode none
## sbatch llm_action/scripts/train.sh --action-version v45 --benchmarks-name dataset_ml --exp-name v45_dataset_ml_dep --max-steps 5 --masking-mode dependencies
## sbatch llm_action/scripts/train.sh --action-version v45 --benchmarks-name dataset_ml --exp-name v45_dataset_ml_graph --max-steps 5 --masking-mode schedule_graph
## sbatch llm_action/scripts/train.sh --action-version v45 --benchmarks-name dataset_ml --exp-name v45_dataset_ml_graph_interm --max-steps 5 --masking-mode schedule_graph --reward-mode intermediate