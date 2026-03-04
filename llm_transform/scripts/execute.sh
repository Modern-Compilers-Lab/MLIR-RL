#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J eval_matmul
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 28
#SBATCH --mem=100G
#SBATCH -t 1-00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate main

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

cd "$(dirname "$(dirname "$(realpath "$0")")")"

echo "Base:"
# TIME_BASE=$(python src/utils/execution.py -t resources/base_schedule.mlir -p resources/base_passes.txt)
# Return saved values since the base doesn't change
case $MATMUL_TYPE in
  1) TIME_BASE=17707650426 ;;
  2) TIME_BASE=346113949 ;;
  3) TIME_BASE=338921024 ;;
  *) echo "Error: MATMUL_TYPE $MATMUL_TYPE does not exist" >&2; exit 1 ;;
esac
echo "Execution time (ns): $TIME_BASE"

echo "Optimized:"
ERR_FILE=$(mktemp)
TIME_OPT=$(
    python src/utils/execution.py $@
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
conda activate torch-cpu
TIME_TORCH=$(python torch_matmul.py $@) 2>"$ERR_FILE"
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
# TIME_TORCH=19321849
echo "Execution time (ns): $TIME_TORCH"
echo "--------------------------"
echo "Speedup over Base: $(echo "scale=4; $TIME_BASE / $TIME_OPT" | bc)x"
echo "Slowdown compared to PyTorch: $(echo "scale=4; $TIME_OPT / $TIME_TORCH" | bc)x"
