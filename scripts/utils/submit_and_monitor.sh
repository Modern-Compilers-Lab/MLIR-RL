#!/bin/bash
#
# Submit a SLURM job and automatically monitor its output
# Usage: ./submit_and_monitor.sh <script_path> [additional_args...]
# Example: ./submit_and_monitor.sh lstm/test_lstm.sh
#          ./submit_and_monitor.sh run_benchmarks.sh lstm 4
#

if [ $# -eq 0 ]; then
    echo "Usage: $0 <script_path> [additional_args...]"
    echo ""
    echo "Examples:"
    echo "  $0 lstm/test_lstm.sh"
    echo "  $0 distilbert/test_distilbert.sh"
    echo "  $0 compare_all.sh"
    echo "  $0 run_benchmarks.sh lstm 4"
    exit 1
fi

SCRIPT_PATH="$1"
shift  # Remove first argument, keep the rest

# Check if script exists
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script not found: $SCRIPT_PATH"
    exit 1
fi

# Extract job name from the script (handles both -J and --job-name forms)
JOB_NAME=$(grep -E "^#SBATCH[[:space:]]+(-J|--job-name)[[:space:]=]" "$SCRIPT_PATH" | head -1 | sed -E 's/.*(-J|--job-name)[[:space:]=]+//')

if [ -z "$JOB_NAME" ]; then
    echo "Error: Could not determine job name from script (no -J or --job-name directive found)"
    exit 1
fi

# Determine log directory (always use absolute path to project root)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="${PROJECT_ROOT}/logs"

# Load MAIL_USER from .env if not already set
if [[ -z "${MAIL_USER:-}" && -f "$PROJECT_ROOT/.env" ]]; then
    MAIL_USER=$(grep -E '^MAIL_USER=' "$PROJECT_ROOT/.env" | cut -d= -f2-)
fi

# Submit job and capture job ID (pass remaining arguments)
echo "Submitting job: $SCRIPT_PATH"
if [ $# -gt 0 ]; then
    echo "With arguments: $@"
fi
echo "----------------------------------------"

SBATCH_OPTS=(--parsable)
if [[ -n "${MAIL_USER:-}" ]]; then
    SBATCH_OPTS+=(--mail-user="$MAIL_USER")
    echo "  Mail notifications → $MAIL_USER"
fi

JOB_OUTPUT=$(sbatch "${SBATCH_OPTS[@]}" "$SCRIPT_PATH" "$@" 2>&1)

if [ $? -ne 0 ]; then
    echo "Error submitting job:"
    echo "$JOB_OUTPUT"
    exit 1
fi

JOB_ID="$JOB_OUTPUT"

# Extract the log file path from #SBATCH --output, then replace %j/%A with the actual job ID
LOG_PATTERN=$(grep -E "^#SBATCH[[:space:]]+(-o|--output)[[:space:]=]" "$SCRIPT_PATH" | head -1 | sed -E "s/.*(-o|--output)[[:space:]=]+//")
LOG_FILE="${LOG_PATTERN//%j/$JOB_ID}"
LOG_FILE="${LOG_FILE//%A/$JOB_ID}"

echo "✓ Job submitted: $JOB_ID"
echo "  Job name: $JOB_NAME"
echo "  Log file: $LOG_FILE"
echo ""
echo "Waiting for log file..."

# Wait for log file to be created (max 60 seconds)
COUNTER=0
while [ ! -f "$LOG_FILE" ] && [ $COUNTER -lt 60 ]; do
    sleep 1
    COUNTER=$((COUNTER + 1))
    
    # Show progress every 10 seconds
    if [ $((COUNTER % 10)) -eq 0 ]; then
        JOB_STATE=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
        if [ -n "$JOB_STATE" ]; then
            echo "  Waiting... (Job state: $JOB_STATE)"
        else
            echo "  Waiting... (Job may have completed)"
            break
        fi
    fi
done

if [ ! -f "$LOG_FILE" ]; then
    echo ""
    echo "Warning: Log file not created after 60 seconds"
    JOB_STATE=$(sacct -j $JOB_ID --format=State --noheader | head -1 | xargs)
    echo "Job state: $JOB_STATE"
    echo ""
    echo "The job may have completed very quickly or failed immediately."
    echo "Check with: cat $LOG_FILE"
    exit 0
fi

echo ""
echo "✓ Log file created, monitoring output..."
echo "=========================================="
echo "(Press Ctrl+C to stop monitoring)"
echo ""

# Monitor the log file
tail -f "$LOG_FILE"