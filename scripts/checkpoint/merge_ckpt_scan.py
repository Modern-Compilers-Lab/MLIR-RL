#!/usr/bin/env python3
"""
merge_ckpt_scan.py
-------------------
Merge per-model checkpoint scan results into a summary.

Usage:
  python scripts/checkpoint/merge_ckpt_scan.py --scan-dir results/full_model/scan --output results/full_model/scan/summary.json
"""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge checkpoint scan results.")
    parser.add_argument("--scan-dir", required=True, help="Directory containing <model>_scan.json files")
    parser.add_argument("--output", required=True, help="Output summary JSON path")
    args = parser.parse_args()

    scan_dir = Path(args.scan_dir)
    results = {}
    per_checkpoint = {}

    for f in sorted(scan_dir.glob("*_scan.json")):
        with open(f) as fh:
            data = json.load(fh)

        model_name = data.get("model", f.stem.replace("_scan", ""))
        results[model_name] = data

        if "checkpoints" in data:
            for ckpt_name, cr in data["checkpoints"].items():
                if ckpt_name not in per_checkpoint:
                    per_checkpoint[ckpt_name] = []
                per_checkpoint[ckpt_name].append({
                    "model": model_name,
                    "speedup": cr["speedup"],
                    "baseline_ns": cr["baseline_ns"],
                    "optimized_ns": cr["optimized_ns"],
                    "blocks_used": cr["blocks_used"],
                    "errors": cr["errors"],
                })

    summary = {
        "models": {},
        "per_checkpoint": {},
        "best_per_model": {},
    }

    for model_name, data in results.items():
        summary["models"][model_name] = {
            "method": data.get("method", ""),
            "total_blocks": data.get("total_blocks_extracted", 0),
            "compute_heavy_blocks": data.get("compute_heavy_blocks", 0),
            "total_baseline_ns": data.get("total_baseline_ns", 0),
        }
        if "checkpoints" in data:
            summary["models"][model_name]["checkpoints"] = {
                k: {"speedup": v["speedup"], "optimized_ns": v["optimized_ns"]}
                for k, v in data["checkpoints"].items()
            }

        best_ckpt = None
        best_speedup = 0
        if "checkpoints" in data:
            for ckpt_name, cr in data["checkpoints"].items():
                if cr["speedup"] > best_speedup:
                    best_speedup = cr["speedup"]
                    best_ckpt = ckpt_name
        summary["best_per_model"][model_name] = {
            "checkpoint": best_ckpt,
            "speedup": best_speedup,
        }

    for ckpt_name, model_results in sorted(per_checkpoint.items()):
        avg_speedup = sum(m["speedup"] for m in model_results) / len(model_results) if model_results else 0
        summary["per_checkpoint"][ckpt_name] = {
            "average_speedup": avg_speedup,
            "models_evaluated": len(model_results),
        }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary → {args.output}")
    print(f"Models: {len(results)}")
    print(f"Checkpoints: {len(per_checkpoint)}")
    print()

    print(f"{'Checkpoint':<20} {'Avg Speedup':>12} {'Models':>8}")
    print("-" * 42)
    for ckpt_name, cr in sorted(summary["per_checkpoint"].items()):
        print(f"{ckpt_name:<20} {cr['average_speedup']:>12.4f}x {cr['models_evaluated']:>8}")

    print()
    print(f"{'Model':<25} {'Best Checkpoint':<20} {'Speedup':>10}")
    print("-" * 57)
    for model_name, best in sorted(summary["best_per_model"].items()):
        print(f"{model_name:<25} {best['checkpoint'] or 'N/A':<20} {best['speedup']:>10.4f}x")


if __name__ == "__main__":
    main()
