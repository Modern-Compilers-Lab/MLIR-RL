#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J claude
#SBATCH -p compute
##SBATCH -q c2
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 7-00
#SBATCH -o logs/claude/%j.log

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
conda activate main

# Execute the code
FULL_SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/{print $2}' | cut -d' ' -f1)
cd "$(dirname "$(dirname "$(realpath "$FULL_SCRIPT_PATH")")")"

# Create a new experiment directory with a unique ID
STATS_DIR="logs/stats"
LAST_ID=$(ls -1 "$STATS_DIR" 2>/dev/null | sort -n | tail -1)
EXPERIMENT_ID=$(( ${LAST_ID:-0} + 1 ))
EXPERIMENT_DIR="$STATS_DIR/$EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"
touch "$EXPERIMENT_DIR/performance.log"
touch "$EXPERIMENT_DIR/tokens.log"
export EXPERIMENT_DIR
echo "Experiment ID: $EXPERIMENT_ID"

# Log token usage from a claude JSON response
log_tokens() {
    local output="$1"
    local duration="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local input_tokens=$(echo "$output" | jq -r '.usage.input_tokens // 0')
    local output_tokens=$(echo "$output" | jq -r '.usage.output_tokens // 0')
    local total_tokens=$(( input_tokens + output_tokens ))
    echo "$timestamp | input=$input_tokens | output=$output_tokens | total=$total_tokens | duration=${duration}s" >> "$EXPERIMENT_DIR/tokens.log"
}

# Start claude code sessions
rm -f logs/jobs/*
START=$(date +%s)
OUTPUT=$(claude --permission-mode dontAsk --print --output-format=json "$(cat resources/prompt.txt)")
DURATION=$(( $(date +%s) - START ))
log_tokens "$OUTPUT" "$DURATION"
# for ((i = 0 ; i < 99 ; i++ )); do
#     START=$(date +%s)
#     OUTPUT=$(claude --continue --print --output-format=json "$(cat resources/prompt.txt)")
#     DURATION=$(( $(date +%s) - START ))
#     log_tokens "$OUTPUT" "$DURATION"
# done

# Create performance plots
python src/tools/plot_performance.py "$EXPERIMENT_ID"

# Cleanup
rm -rf tmp/*
