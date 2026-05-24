#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J claude
#SBATCH -p compute
#SBATCH -q c2
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

# Optional: subset of benchmarks/instances to optimize (names or full IDs).
# Example: sbatch scripts/claude.sh matmul_2 conv_2d
INSTANCE_FILTER="$*"
export INSTANCE_FILTER
echo "Instance filter: ${INSTANCE_FILTER:-ALL}"

# Create a new experiment directory with a unique ID
STATS_DIR="logs/stats"
LAST_ID=$(ls -1 "$STATS_DIR" 2>/dev/null | sort -n | tail -1)
EXPERIMENT_ID=$(( ${LAST_ID:-0} + 1 ))
EXPERIMENT_DIR="$STATS_DIR/$EXPERIMENT_ID"
mkdir -p "$EXPERIMENT_DIR"
touch "$EXPERIMENT_DIR/claude_optimization.log"
touch "$EXPERIMENT_DIR/performance.log"
touch "$EXPERIMENT_DIR/tokens.log"
export EXPERIMENT_DIR
echo "Experiment ID: $EXPERIMENT_ID"

TOTAL_INPUT_TOKENS=0
TOTAL_OUTPUT_TOKENS=0

# Format a duration in seconds into a human-readable string
format_duration() {
    local s="$1"
    if (( s < 60 )); then
        printf '%ds' "$s"
    elif (( s < 3600 )); then
        printf '%dm%02ds' $(( s / 60 )) $(( s % 60 ))
    elif (( s < 86400 )); then
        printf '%dh%02dm%02ds' $(( s / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
    else
        printf '%dd%02dh%02dm%02ds' $(( s / 86400 )) $(( (s % 86400) / 3600 )) $(( (s % 3600) / 60 )) $(( s % 60 ))
    fi
}

# Log token usage from a claude JSON response
log_tokens() {
    local output="$1"
    local duration="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local input_tokens=$(echo "$output" | jq -r '.usage.input_tokens // 0')
    local output_tokens=$(echo "$output" | jq -r '.usage.output_tokens // 0')
    local total_tokens=$(( input_tokens + output_tokens ))
    TOTAL_INPUT_TOKENS=$(( TOTAL_INPUT_TOKENS + input_tokens ))
    TOTAL_OUTPUT_TOKENS=$(( TOTAL_OUTPUT_TOKENS + output_tokens ))
    echo "$timestamp | input=$input_tokens | output=$output_tokens | total=$total_tokens | duration=$(format_duration "$duration")" >> "$EXPERIMENT_DIR/tokens.log"
}

# Render the prompt; scope is derived from $INSTANCE_FILTER.
CLAUDE_PROMPT=$(python -m llm_transform.tools.build_prompt)

# Start claude code sessions
rm -f logs/jobs/*
EXPERIMENT_START=$(date +%s)
START=$(date +%s)
OUTPUT=$(claude --effort max --permission-mode dontAsk --print --output-format=json "$CLAUDE_PROMPT")
DURATION=$(( $(date +%s) - START ))
log_tokens "$OUTPUT" "$DURATION"
for ((i = 1 ; i < 5 ; i++ )); do
    START=$(date +%s)
    OUTPUT=$(claude --continue --effort max --permission-mode dontAsk --print --output-format=json "$CLAUDE_PROMPT")
    DURATION=$(( $(date +%s) - START ))
    log_tokens "$OUTPUT" "$DURATION"
done
TOTAL_DURATION=$(( $(date +%s) - EXPERIMENT_START ))
TOTAL_TOKENS=$(( TOTAL_INPUT_TOKENS + TOTAL_OUTPUT_TOKENS ))
echo "TOTAL | input=$TOTAL_INPUT_TOKENS | output=$TOTAL_OUTPUT_TOKENS | total=$TOTAL_TOKENS | duration=$(format_duration "$TOTAL_DURATION")" >> "$EXPERIMENT_DIR/tokens.log"

# Create performance plots
python -m llm_transform.tools.plot_performance "$EXPERIMENT_ID"

# Cleanup
rm -rf tmp/*
