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

# Parse arguments
BUFFERIZE=true

# Loop through arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -no-bufferize)
      BUFFERIZE=false
      shift
      ;;
    -[0-9]*)
      MATMUL_TYPE="${1:1}"
      shift
      ;;
    *)
      # Assume any other argument is the SCHED_NAME
      SCHED_NAME=$1
      shift
      ;;
  esac
done

# Check if SCHED_NAME was provided
if [ -z "$SCHED_NAME" ]; then
    echo "Error: SCHED_NAME variable is missing."
    echo "Usage: sbatch run.sh [-bufferize] <SCHED_NAME>"
    exit 1
fi

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
export KMP_BLOCKTIME=0

# Execute the code
set -eo pipefail
INPUT_SRC="matmul_${MATMUL_TYPE}.mlir"

echo "Base:"
# TIME_BASE=$(mlir-opt ${INPUT_SRC} | python run.py -p base.txt)
# Return saved values since the base doesn't change
case $MATMUL_TYPE in
  1) TIME_BASE=17707650426 ;;
  2) TIME_BASE=346113949 ;;
  3) TIME_BASE=338921024 ;;
  *) echo "Error: MATMUL_TYPE $MATMUL_TYPE does not exist" >&2; exit 1 ;;
esac
echo "Execution time (ns): $TIME_BASE"

echo "Optimized:"
TRANSFORM_CMD=(
  mlir-opt
  -transform-preload-library="transform-library-paths=schedules/${SCHED_NAME}_${MATMUL_TYPE}.mlir"
  -transform-interpreter
)
ERR_FILE=$(mktemp)
TIME_OPT=$(
  if [ "$BUFFERIZE" = true ]; then
    mlir-opt "${INPUT_SRC}" \
    -eliminate-empty-tensors -empty-tensor-to-alloc-tensor \
    -one-shot-bufferize="unknown-type-conversion=identity-layout-map function-boundary-type-conversion=identity-layout-map bufferize-function-boundaries" \
    -buffer-results-to-out-params="hoist-static-allocs add-result-attr" \
    -canonicalize -cse | "${TRANSFORM_CMD[@]}" -
  else
    "${TRANSFORM_CMD[@]}" "${INPUT_SRC}"
  fi | python run.py -p schedules/${SCHED_NAME}_${MATMUL_TYPE}.txt
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
TIME_TORCH=$(python torch_matmul.py ${MATMUL_TYPE})
re='^[0-9]+$'
if ! [[ $TIME_TORCH =~ $re ]]; then
   echo "Error: TIME_TORCH is not a number" >&2; exit 1
fi
# TIME_TORCH=19321849
echo "Execution time (ns): $TIME_TORCH"
echo "--------------------------"
echo "Speedup over Base: $(echo "scale=4; $TIME_BASE / $TIME_OPT" | bc)x"
echo "Slowdown compared to PyTorch: $(echo "scale=4; $TIME_OPT / $TIME_TORCH" | bc)x"
