#!/bin/bash
# Merge chunked full-model optimization results into a single JSON + CSV summary.
#
# Usage:
#   bash scripts/merge_full_model_results.sh [chunks_dir] [output_prefix]
#
# Defaults:
#   chunks_dir   = results/full_model
#   output_prefix = results/full_model/merged

set -e

CHUNKS_DIR="${1:-results/full_model}"
OUTPUT_PREFIX="${2:-$CHUNKS_DIR/merged}"

echo "Merging chunk files from $CHUNKS_DIR ..."

# Collect all chunk files
CHUNKS=($(ls "$CHUNKS_DIR"/chunk_*.json 2>/dev/null | sort -t_ -k2 -n))

if [[ ${#CHUNKS[@]} -eq 0 ]]; then
    echo "ERROR: No chunk_*.json files found in $CHUNKS_DIR"
    exit 1
fi

echo "Found ${#CHUNKS[@]} chunk(s)."

# Merge into a single JSON
python3 - "$OUTPUT_PREFIX" "${CHUNKS[@]}" << 'PYEOF'
import sys, json

output_prefix = sys.argv[1]
chunk_files = sys.argv[2:]

merged = {}
for cf in chunk_files:
    try:
        with open(cf) as f:
            data = json.load(f)
        merged.update(data)
        print(f"  + {cf} ({len(data)} entries)")
    except Exception as e:
        print(f"  SKIP {cf}: {e}")

# Write merged JSON
json_path = output_prefix + ".json"
with open(json_path, "w") as f:
    json.dump(merged, f, indent=2)
print(f"\nMerged {len(merged)} models → {json_path}")

# Write CSV summary
csv_path = output_prefix + ".csv"
with open(csv_path, "w") as f:
    f.write("Model,Baseline(ns),Optimized(ns),Speedup\n")
    for name in sorted(merged.keys()):
        d = merged[name]
        if "error" in d:
            f.write(f'{name},ERROR: {d["error"]},,\n')
        else:
            f.write(f'{name},{d.get("baseline_ns",0)},{d.get("optimized_ns",0)},{d.get("speedup",1.0):.4f}\n')
print(f"CSV summary → {csv_path}")

# Print summary table
print(f"\n{'Model':<25} {'Baseline (ns)':>16} {'Optimized (ns)':>16} {'Speedup':>10}")
print("-" * 69)
for name in sorted(merged.keys()):
    d = merged[name]
    if "error" in d:
        print(f"{name:<25} {'ERROR: ' + d['error'][:30]}")
    else:
        bl = d.get("baseline_ns", 0)
        op = d.get("optimized_ns", 0)
        su = d.get("speedup", 1.0)
        print(f"{name:<25} {bl:>16,} {op:>16,} {su:>10.4f}x")
PYEOF

echo ""
echo "Merge complete."
echo "  JSON: ${OUTPUT_PREFIX}.json"
echo "  CSV:  ${OUTPUT_PREFIX}.csv"
