#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J execute
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH --mem=100G
#SBATCH -t 1-00
#SBATCH -o logs/jobs/%x_%j.out
#SBATCH -e logs/jobs/%x_%j.err

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Environment variables
export OMP_NUM_THREADS=$(nproc)
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_SCHEDULE=static
export OMP_DYNAMIC=FALSE
export OMP_WAIT_POLICY=passive
export KMP_BLOCKTIME=infinite

# Execute the code
set -eo pipefail
FULL_SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | cut -d' ' -f1)
cd "$(dirname "$(dirname "$(realpath "$FULL_SCRIPT_PATH")")")"

# Load the conda environment name (MAIN_ENV) and activate
if [ ! -f scripts/env.local.sh ]; then
    echo "Error: scripts/env.local.sh not found. Copy scripts/env.local.sh.example to scripts/env.local.sh and set MAIN_ENV." >&2
    exit 1
fi
source scripts/env.local.sh
conda activate "$MAIN_ENV"

orig_args=("$@")
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -i|--id) CODE_ID="$2";;
        --id=*) CODE_ID="${1#--id=}";;
    esac
    shift
done
if [ -z "$CODE_ID" ]; then
    echo "Error: Missing required argument -i or --id" >&2
    exit 1
fi
set -- "${orig_args[@]}"
echo "Evaluating code $CODE_ID:"

# echo "Base:"
# TIME_BASE=$(python -m llm_transform.utils.execution -i $CODE_ID -t resources/base_schedule.mlir -p resources/base_passes.txt)
# echo "Execution time (ns): $TIME_BASE"

echo "Optimized:"
ERR_FILE=$(mktemp)
TIME_OPT=$(
    python -m llm_transform.utils.execution $@
) 2>"$ERR_FILE"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error: Optimized execution failed (exit code $EXIT_CODE):" >&2
  cat "$ERR_FILE" >&2
  rm -f "$ERR_FILE"
  exit 1
fi
re='^[0-9]+$'
if ! [[ $TIME_OPT =~ $re ]]; then
  echo "Error: TIME_OPT is not a number: '$TIME_OPT'" >&2
  cat "$ERR_FILE" >&2
  rm -f "$ERR_FILE"
  exit 1
fi
rm -f "$ERR_FILE"
echo "Execution time (ns): $TIME_OPT"

echo "PyTorch:"
TIME_TORCH=$(python -m llm_transform.torch_exec $CODE_ID) 2>"$ERR_FILE"
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Error: PyTorch execution failed (exit code $EXIT_CODE):" >&2
  cat "$ERR_FILE" >&2
  rm -f "$ERR_FILE"
  exit 1
fi
re='^[0-9]+$'
if ! [[ $TIME_TORCH =~ $re ]]; then
  echo "Error: TIME_TORCH is not a number" >&2; exit 1
fi
echo "Execution time (ns): $TIME_TORCH"
echo "--------------------------"
# echo "Speedup over Base: $(echo "scale=4; $TIME_BASE / $TIME_OPT" | bc)x"
SPEEDUP=$(echo "scale=4; $TIME_TORCH / $TIME_OPT" | bc)
echo "Speedup compared to PyTorch: ${SPEEDUP}x"

# Log performance metrics to the experiment directory if it exists
if [ -n "$EXPERIMENT_DIR" ]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') | id=$CODE_ID | speedup=${SPEEDUP}x | exec_time_ns=$TIME_OPT" >> "$EXPERIMENT_DIR/performance.log"
fi
