#!/usr/bin/env python3
"""
Classify all benchmarks for plotting.
Output: plots/benchmark_classification.csv
Columns: benchmark, eval_set, category, model_family, full_model, op_type
"""

import os, re, json, csv

EVAL_DIR = "data/new_dataset/all/eval"
EVAL_FULL_DIR = "data/new_dataset/all/eval_full"
EVAL_BASE_JSON = "results/new_dataset_results/baselines/mlir/eval_base.json"
EVAL_FULL_JSON = "results/new_dataset_results/baselines/mlir/eval_full_base.json"
OUT = "plots/benchmark_classification.csv"

# Load benchmark lists
eval_sets = {
    "eval_base": (EVAL_DIR, EVAL_BASE_JSON),
    "eval_full": (EVAL_FULL_DIR, EVAL_FULL_JSON),
}

# Known legacy model-to-full-model mapping
LEGACY_TO_FULL = {
    "albert": "albert",
    "bart": "bart",
    "bert": "bert",
    "convnext_tiny": "convnext_tiny",
    "densen121": "densenet121",
    "deberta": "deberta",
    "distilbert": "distilbert",
    "efficientnet_b0": "efficientnet_b0",
    "gat": "gat",
    "gcn": "gcn",
    "gin": "gin",
    "gpt2": "gpt2",
    "gpt2-large": "gpt2",
    "gpt2-medium": "gpt2",
    "llama3_2_1b": "llama3_2_1b",
    "lstm": "lstm",
    "mobilenet_v3_small": "mobilenet_v3_small",
    "resnet18": "resnet18",
    "resnet50": "resnet50",
    "resnext50": "resnext50",
    "roberta": "roberta",
    "t5": "t5",
    "vgg11": "vgg11",
    "vgg16": "vgg16",
    "vit_b_16": "vit_b_16",
    "whisper_base": "whisper_base",
    "yolov8m": "yolov8m",
}


def extract_model_name(name):
    """Extract model name from benchmark name."""
    for key, full in sorted(LEGACY_TO_FULL.items(), key=lambda x: -len(x[0])):
        if name.startswith(key + "_") or name == key:
            return key, full
    return None, None


def classify_op_from_file(filepath):
    """Classify a single_bench or bench file by reading its .mlir content."""
    try:
        with open(filepath) as fh:
            content = fh.read()
    except (OSError, IOError):
        return "unknown"
    ops = set(re.findall(r'linalg\.(\w+(?:_\w+)*)', content))
    ops.discard("yield")
    ops.discard("index")
    if not ops:
        return "unknown"
    ops_str = ",".join(sorted(ops))
    # Prioritize specific ops
    if "matmul" in ops:
        return "matmul"
    if "batch_matmul" in ops:
        return "batch_matmul"
    if "conv_2d" in ops_str:
        return "conv2d"
    if "pooling" in ops_str:
        return "pooling"
    if "generic" in ops:
        return "generic"
    return "mixed"


def classify_op_from_name(name):
    """Determine op_type from benchmark name suffix for model benchmarks."""
    if "_block_" in name or (name.split("_")[-1].isdigit() and "_block" in name):
        return "block"
    if "_batch_matmul_" in name or "_batch_matmul" in name.rsplit("_", 1)[0]:
        return "batch_matmul"
    if "_conv_2d_" in name:
        return "conv2d"
    if "_matmul_" in name or "_matmul" in name.rsplit("_", 1)[0]:
        return "matmul"
    if "_generic_" in name or "_generic" in name.rsplit("_", 1)[0]:
        return "generic"
    if "_pooling_" in name or "_pooling" in name.rsplit("_", 1)[0]:
        return "pooling"
    return "unknown"


def classify(bench_name, eval_set_name, bench_dir):
    """Return (category, model_family, full_model, op_type)."""
    filepath = os.path.join(bench_dir, bench_name + ".mlir")

    # Legacy single_bench
    if bench_name.startswith("single_bench_"):
        op = classify_op_from_file(filepath)
        return ("legacy_single", "legacy", "synthetic", op)

    # Legacy bench_
    if bench_name.startswith("bench_"):
        op = classify_op_from_file(filepath)
        return ("legacy_block", "legacy", "synthetic", op)

    # Model benchmarks
    model_name, full_model = extract_model_name(bench_name)
    if model_name is None:
        # Fallback: try to parse from the name
        parts = bench_name.split("_")
        if len(parts) >= 2:
            fallback = parts[0]
            op = classify_op_from_name(bench_name)
            return ("model_single_op", fallback, fallback, op)
        return ("unknown", "unknown", "unknown", "unknown")

    op_type = classify_op_from_name(bench_name)
    if op_type == "block":
        category = "model_block"
    else:
        category = "model_single_op"

    return (category, model_name, full_model, op_type)


def main():
    rows = []
    for set_name, (bench_dir, json_path) in eval_sets.items():
        with open(json_path) as f:
            benchmarks = json.load(f)
        for bench_name in benchmarks:
            category, model, full, op_type = classify(bench_name, set_name, bench_dir)
            rows.append({
                "benchmark": bench_name,
                "eval_set": set_name,
                "category": category,
                "model_family": model,
                "full_model": full,
                "op_type": op_type,
            })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["benchmark", "eval_set", "category", "model_family", "full_model", "op_type"])
        w.writeheader()
        w.writerows(rows)

    # Summary
    from collections import Counter
    cats = Counter(r["category"] for r in rows)
    print(f"Written {len(rows)} rows to {OUT}")
    print("\nCategory distribution:")
    for cat, count in cats.most_common():
        print(f"  {cat}: {count}")

    # Show op_type breakdown
    for cat in ["model_single_op", "legacy_single", "legacy_block"]:
        ops = Counter(r["op_type"] for r in rows if r["category"] == cat)
        if ops:
            print(f"\n{cat} op_types: {dict(ops)}")

    # Show model distribution
    models = Counter(r["full_model"] for r in rows if r["category"] in ("model_block", "model_single_op"))
    print(f"\nModel distribution (blocks+single_ops): {dict(models)}")


if __name__ == "__main__":
    main()
