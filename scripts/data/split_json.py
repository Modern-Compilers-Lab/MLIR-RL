"""Split a base execution times JSON into train and eval files.

Uses stratified splitting by model family (derived from benchmark name prefix)
so that every model is proportionally represented in both sets.

Usage:
    python scripts/data/split_json.py --input data/all_base_exec_times.json
    python scripts/data/split_json.py --input data/all_base_exec_times.json --eval-ratio 0.15 --seed 123

Output files are written next to the input file:
    <stem>_train.json
    <stem>_eval.json

Only entries with exec_time > 0 (i.e. not failed) are included.
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from utils.implementation import get_autoschedular_impl, get_base_file_path

parser = argparse.ArgumentParser()
parser.add_argument("config_path", nargs="?", default=None, help="Path to config JSON (positional alternative to --config)")
parser.add_argument("implementation_positional", nargs="?", default=None, help="Autoscheduler implementation package (positional alternative to --implementation)")
parser.add_argument("--config", default=None, help="Path to config JSON (derives input path)")
parser.add_argument("--input", default=None, help="Override: path to base exec times JSON {bench_name: ns}")
parser.add_argument("--implementation", default=None, help="Autoscheduler implementation package (default: AUTOSCHEDULER_IMPL or rl_autoschedular)")
parser.add_argument("--eval-ratio", type=float, default=0.2, help="Fraction reserved for eval (default: 0.2)")
parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
args = parser.parse_args()

config_path = args.config or args.config_path
implementation = args.implementation or args.implementation_positional or get_autoschedular_impl(config_path=config_path)

if config_path:
    with open(config_path) as _f:
        _cfg = json.load(_f)
    # Default to the dataset-level generic base.json (shared across implementations).
    input_path = args.input or str(get_base_file_path(_cfg["results_dir"], implementation=None))
else:
    if not args.input:
        parser.error("Provide --config or --input")
    input_path = args.input
with open(input_path) as f:
    data: dict[str, int] = json.load(f)

# Drop failed entries
valid = {k: v for k, v in data.items() if v > 0}
failed = len(data) - len(valid)
if failed:
    print(f"Dropped {failed} failed entries (exec_time <= 0)")

KNOWN_MODELS = [
    "albert", "bart", "bert", "convnext_tiny", "densen121", "deberta", "distilbert",
    "efficientnet_b0", "gat", "gcn", "gin", "gpt2", "gpt2-large", "gpt2-medium",
    "llama3_2_1b", "lstm", "mobilenet_v3_small", "resnet18", "resnet50", "resnext50",
    "roberta", "t5", "vgg11", "vgg16", "vit_b_16", "whisper_base", "yolov8m",
]

# Group by model family: take longest prefix before first op-type keyword.
# Benchmark names look like: albert_generic_0, densenet121_sz192_bs4_conv_0, etc.
# For single_ops: albert_add_17, bart_batch_matmul_5, add_14_150_15_28, etc.
OP_KEYWORDS = {"generic", "matmul", "batch_matmul", "conv", "conv_2d", "pooling", "relu",
               "add", "mul", "sub", "reduce_sum", "block", "patterns_bench", "pattern",
               "residual_bench", "resnet_bench"}

def model_family(name: str) -> str:
    """Extract model family from benchmark name.

    First checks known model prefixes (longest match first).
    Then falls back to keyword-based extraction for new_dataset style names.
    Synthetic benchmarks (no model prefix) are grouped as 'synthetic'.
    """
    for m in sorted(KNOWN_MODELS, key=len, reverse=True):
        if name.startswith(m + "_"):
            return m

    parts = name.split("_")
    for i, part in enumerate(parts):
        if part in OP_KEYWORDS:
            return "_".join(parts[:i]) or "synthetic"
        if part.startswith("sz") and part[2:].isdigit():
            return "_".join(parts[:i]) or "synthetic"
        # bench_NNNN — group all numeric suffixes under "bench"
        if part.isdigit():
            prefix = "_".join(parts[:i]) or "synthetic"
            if prefix == "bench":
                return "bench"
    return "synthetic"

# Stratified split: sample eval_ratio from each family
random.seed(args.seed)
groups: dict[str, list[str]] = defaultdict(list)
for k in valid:
    groups[model_family(k)].append(k)

train_keys: list[str] = []
eval_keys: list[str] = []

for family, keys in sorted(groups.items()):
    random.shuffle(keys)
    n_eval = max(1, round(len(keys) * args.eval_ratio))
    eval_keys.extend(keys[:n_eval])
    train_keys.extend(keys[n_eval:])
    print(f"  {family:<30s}  total={len(keys):4d}  train={len(keys)-n_eval:4d}  eval={n_eval:4d}")

train_data = {k: valid[k] for k in train_keys}
eval_data  = {k: valid[k] for k in eval_keys}

stem = os.path.splitext(input_path)[0]
train_path = stem + "_train.json"
eval_path  = stem + "_eval.json"

with open(train_path, "w") as f:
    json.dump(train_data, f, indent=4)
with open(eval_path, "w") as f:
    json.dump(eval_data, f, indent=4)

print(f"\nTotal valid: {len(valid)}")
print(f"Train: {len(train_data)} entries  →  {train_path}")
print(f"Eval:  {len(eval_data)} entries  →  {eval_path}")
