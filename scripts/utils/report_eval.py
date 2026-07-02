#!/usr/bin/env python3
"""Report evaluation progression from eval/ checkpoint files.

Usage:
  python scripts/utils/report_eval.py                     # all agents, all checkpoints
  python scripts/utils/report_eval.py -v v4_6 v4_7        # specific agents
  python scripts/utils/report_eval.py --missing            # show missing evals
  python scripts/utils/report_eval.py --best               # show only best per agent
  python scripts/utils/report_eval.py --corrupted          # show corrupted checkpoints
  python scripts/utils/report_eval.py -c 500               # show only checkpoint 500
  python scripts/utils/report_eval.py --json out.json      # export results as JSON
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_DIRS = {
    "new": "results/new_dataset_results",
    "single_ops": "results/single_ops_dataset_results",
}

DATASET_BASELINES = {
    "new": "results/new_dataset_results/baselines/mlir/eval_base.json",
    "single_ops": "results/single_ops_dataset_results/baselines/mlir/base_eval.json",
}

AGENT_REGISTRY = {
    "v0": {"results_dir": "results/new_dataset_results/v0_agent", "display": "V0"},
    "v4_5": {"results_dir": "results/new_dataset_results/v4_5_agent", "display": "V4.5"},
    "v4_6": {"results_dir": "results/new_dataset_results/v4_6_agent", "display": "V4.6"},
    "v4_7": {"results_dir": "results/new_dataset_results/v4_7_agent", "display": "V4.7"},
    "v4_8": {"results_dir": "results/new_dataset_results/v4_8_agent", "display": "V4.8"},
    "no_hw": {"results_dir": "results/new_dataset_results/ablation_study/v45_no_hw_agent", "display": "No-HW"},
    "no_shaped_reward": {"results_dir": "results/new_dataset_results/ablation_study/v45_no_shaped_reward_agent", "display": "No-ShapedRwd"},
    "no_transformer": {"results_dir": "results/new_dataset_results/ablation_study/v45_no_transformer_agent", "display": "No-Transformer"},
}

BASELINE_PATH = "results/new_dataset_results/baselines/mlir/eval_base.json"


def build_agent_registry(dataset: str):
    """Build agent registry for a specific dataset."""
    base = DATASET_DIRS.get(dataset, DATASET_DIRS["new"])
    return {
        "v0": {"results_dir": f"{base}/v0_agent", "display": "V0"},
        "v4_6": {"results_dir": f"{base}/v4_6_agent", "display": "V4.6"},
        "v4_7": {"results_dir": f"{base}/v4_7_agent", "display": "V4.7"},
        "v4_8": {"results_dir": f"{base}/v4_8_agent", "display": "V4.8"},
        "v4_9_small": {"results_dir": f"{base}/v4_9_small_agent", "display": "V4.9-S"},
        "v4_9_large": {"results_dir": f"{base}/v4_9_large_agent", "display": "V4.9-L"},
    }


def load_baseline():
    """Load baseline execution times."""
    path = os.path.join(PROJECT_ROOT, BASELINE_PATH)
    if not os.path.isfile(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_baseline_from_path(path: str):
    """Load baseline execution times from a specific path."""
    full_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
    if not os.path.isfile(full_path):
        return {}
    with open(full_path) as f:
        return json.load(f)


def get_checkpoint_files(eval_dir: str) -> list[str]:
    """List checkpoint_*.json files sorted by checkpoint number."""
    if not os.path.isdir(eval_dir):
        return []
    files = [f for f in os.listdir(eval_dir)
             if f.startswith("checkpoint_") and f.endswith(".json")]
    files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    return files


def get_model_checkpoint_indices(models_dir: str) -> set[int]:
    """Get set of checkpoint indices from model files."""
    if not os.path.isdir(models_dir):
        return set()
    indices = set()
    for f in os.listdir(models_dir):
        m = re.match(r"model_(\d+)\.pt", f)
        if m:
            idx = int(m.group(1))
            if idx >= 50 and idx % 50 == 0:
                indices.add(idx)
    return indices


def compute_speedups(eval_json: dict, baseline: dict):
    """Compute speedups from eval exec times vs baseline.
    Returns: (speedups list, failed count, total count)
    """
    speedups = []
    failed = 0
    total = 0
    for bench_name, opt_ns in eval_json.items():
        total += 1
        base_ns = baseline.get(bench_name)
        if base_ns is None or base_ns <= 0:
            failed += 1
        elif opt_ns is None or opt_ns <= 0:
            failed += 1
        else:
            speedups.append(base_ns / opt_ns)
    return speedups, failed, total


def stats(speedups: list[float]):
    """Compute summary statistics."""
    if not speedups:
        return {"arith_mean": 0, "geo_mean": 0, "best": 0, "worst": 0, "n": 0}
    n = len(speedups)
    arith = sum(speedups) / n
    try:
        geo = math.exp(sum(math.log(max(s, 1e-12)) for s in speedups) / n)
    except (ValueError, OverflowError):
        geo = 0.0
    return {
        "arith_mean": arith,
        "geo_mean": geo,
        "best": max(speedups),
        "worst": min(speedups),
        "n": n,
    }


def print_table(rows: list[dict], columns: list[str]):
    """Print a formatted table."""
    col_map = {
        "agent": "Agent",
        "checkpoint": "CP",
        "benchmarks": "Benchs",
        "failed": "Failed",
        "arith": "ArithAvg",
        "geo": "GeoAvg",
        "best": "Best",
        "worst": "Worst",
    }
    header = [col_map.get(c, c) for c in columns]
    rows_out = [header]

    for row in rows:
        r = []
        for c in columns:
            val = row.get(c, "")
            if isinstance(val, float):
                val = f"{val:.2f}x"
            r.append(str(val))
        rows_out.append(r)

    col_widths = [max(len(str(row[i])) for row in rows_out) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)

    print(fmt.format(*header))
    print("-" * (sum(col_widths) + 2 * (len(col_widths) - 1)))
    for row in rows_out[1:]:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(description="Report evaluation progression")
    parser.add_argument("-v", "--agents", nargs="*",
                        help="Agent keys (e.g. v4_6 v4_7)")
    parser.add_argument("-d", "--dataset", choices=["new", "single_ops"], default="new",
                        help="Dataset to report (default: new)")
    parser.add_argument("--missing", action="store_true",
                        help="Show checkpoints that have models but no eval")
    parser.add_argument("--best", action="store_true",
                        help="Show only the best checkpoint per agent")
    parser.add_argument("--min-benchs", type=int, default=100,
                        help="Minimum benchmarks for --best mode (default: 100)")
    parser.add_argument("--corrupted", action="store_true",
                        help="Show checkpoints with fewer than --corrupted-threshold benchmarks")
    parser.add_argument("--corrupted-threshold", type=int, default=2000,
                        help="Benchmark count threshold for corruption detection (default: 2000)")
    parser.add_argument("-c", "--checkpoint", type=int,
                        help="Show only a specific checkpoint (e.g. -c 500)")
    parser.add_argument("--json", metavar="FILE",
                        help="Export results as JSON to file")
    args = parser.parse_args()

    agent_registry = build_agent_registry(args.dataset)
    baseline_path = DATASET_BASELINES.get(args.dataset, DATASET_BASELINES["new"])

    agents = args.agents if args.agents else sorted(agent_registry.keys())

    baseline = load_baseline_from_path(baseline_path)
    if baseline:
        print(f"Baseline: {len(baseline)} benchmarks ({baseline_path})")
    else:
        print(f"Warning: Baseline not found at {baseline_path}, showing raw exec times")

    all_rows = []
    missing_list = []

    for agent_key in agents:
        reg = agent_registry.get(agent_key)
        if not reg:
            continue
        agent_dir = os.path.join(PROJECT_ROOT, reg["results_dir"])
        eval_dir = os.path.join(agent_dir, "eval")
        models_dir = os.path.join(agent_dir, "models")

        ckpt_files = get_checkpoint_files(eval_dir)

        if args.checkpoint:
            target = f"checkpoint_{args.checkpoint}.json"
            ckpt_files = [f for f in ckpt_files if f == target]

        for cf in ckpt_files:
            ckpt_num = int(cf.split("_")[1].split(".")[0])
            fpath = os.path.join(eval_dir, cf)

            try:
                with open(fpath) as f:
                    eval_data = json.load(f)
            except Exception:
                continue

            speedups, failed, total = compute_speedups(eval_data, baseline)
            s = stats(speedups)

            all_rows.append({
                "agent": f"{reg['display']}",
                "checkpoint": ckpt_num,
                    "benchmarks": s["n"],
                    "failed": failed,
                    "arith": s["arith_mean"],
                    "geo": s["geo_mean"],
                    "best": s["best"],
                    "worst": s["worst"],
                })

            # Check missing: models exist but no eval
            if args.missing:
                model_ckpts = get_model_checkpoint_indices(models_dir)
                eval_ckpts = {int(f.split("_")[1].split(".")[0]) for f in ckpt_files}
                missing = sorted(model_ckpts - eval_ckpts)
                for ckpt in missing:
                    missing_list.append({
                        "agent": f"{reg['display']}",
                        "checkpoint": ckpt,
                        "status": "missing eval",
                    })

    if args.best and all_rows:
        # Filter out incomplete evals (too few benchmarks)
        sufficient = [r for r in all_rows if r["benchmarks"] >= args.min_benchs]
        if not sufficient:
            sufficient = all_rows  # fallback if all are partial
        # Group by agent, pick best by geo_mean
        best_by_agent = {}
        for row in sufficient:
            key = row["agent"]
            if key not in best_by_agent or row["geo"] > best_by_agent[key]["geo"]:
                best_by_agent[key] = row
        all_rows = list(best_by_agent.values())

    # Corrupted checkpoint detection
    corrupted_rows = [r for r in all_rows if r["benchmarks"] < args.corrupted_threshold]

    if all_rows:
        columns = ["agent", "checkpoint", "benchmarks", "failed", "arith", "geo", "best", "worst"]
        print_table(all_rows, columns)
        print(f"\nTotal: {len(all_rows)} checkpoint evals")

    if args.corrupted and corrupted_rows:
        print(f"\n=== Potentially Corrupted Checkpoints ({len(corrupted_rows)}) ===")
        print(f"  (fewer than {args.corrupted_threshold} benchmarks)\n")
        columns = ["agent", "checkpoint", "benchmarks", "failed", "arith", "geo"]
        print_table(corrupted_rows, columns)

    if args.missing and missing_list:
        print(f"\n=== Missing Evaluations ({len(missing_list)}) ===")
        columns = ["agent", "checkpoint", "status"]
        print_table(missing_list, columns)

    if args.json and all_rows:
        export = {
            "checkpoints": all_rows,
            "corrupted": corrupted_rows,
            "missing": missing_list,
        }
        with open(args.json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\nExported to {args.json}")

    if not all_rows and not missing_list:
        print("No evaluation data found.")


if __name__ == "__main__":
    main()
