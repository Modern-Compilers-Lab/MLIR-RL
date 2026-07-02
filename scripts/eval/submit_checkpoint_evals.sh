#!/bin/bash
# Submit checkpoint evaluations for V4.6, V4.7, V4.8
# Evaluates checkpoints 100, 200, 300, 400, 500 on each version
# Total: 15 jobs, ~2h each on 2163 benchmarks

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Submitting 15 checkpoint eval jobs..."
echo ""

for version in v4_6 v4_7 v4_8; do
    for ckpt in 100 200 300 400 500; do
        CONFIG="$PROJECT_ROOT/config/new_dataset/eval/${version}_eval.json"
        if [[ ! -f "$CONFIG" ]]; then
            echo "ERROR: Config not found: $CONFIG"
            continue
        fi
        sbatch "$PROJECT_ROOT/scripts/eval/eval.sh" "$CONFIG" --checkpoint "$ckpt"
        sleep 3
    done
done

echo ""
echo "All jobs submitted. Check: squeue -u mb10856"