"""
build_checkpoint_comparison.py
-------------------------------
Merge full-model + block results into a checkpoint comparison table.
Reads:
  - results/full_model/merged.json (full-model successes, checkpoint 715)
  - results/full_model/blocks_v10_*.json (block-based v10: 791, 1500, 1983)
Writes:
  - results/full_model/checkpoint_comparison.json
  - results/full_model/checkpoint_comparison.csv
"""

import json
import csv
from pathlib import Path

RESULTS_DIR = Path("results/full_model")

# All models in display order
ALL_MODELS = [
    "albert", "bart", "bert", "convnext_tiny", "deberta",
    "densenet121", "distilbert", "efficientnet_b0", "gat", "gcn",
    "gpt2", "lstm", "mobilenet_v3_small", "resnet18", "resnet50",
    "resnext50", "roberta", "t5", "vgg11", "vit_b_16",
]

CHECKPOINTS = ["model_715", "model_791", "model_1500", "model_1983"]

def load_merged():
    path = RESULTS_DIR / "merged.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)

def load_v10_blocks():
    results = {}
    for p in sorted(RESULTS_DIR.glob("blocks_v10_*.json")):
        model = p.stem.replace("blocks_v10_", "")
        with open(p) as f:
            data = json.load(f)
        results[model] = data
    return results

def main():
    merged = load_merged()
    v10 = load_v10_blocks()

    comparison = {}
    for model in ALL_MODELS:
        entry = {"model": model, "method": "", "baseline_ns": 0}
        checkpoints = {}

        # Full-model checkpoint 715
        if model in merged and "speedup" in merged.get(model, {}):
            m = merged[model]
            if not isinstance(m, dict) or "error" in m:
                checkpoints["model_715"] = {
                    "method": "full_model",
                    "baseline_ns": m.get("baseline_ns", 0),
                    "optimized_ns": m.get("optimized_ns", 0),
                    "speedup": m.get("speedup", 1.0),
                    "num_ops": m.get("num_ops", 0),
                }
            else:
                checkpoints["model_715"] = {
                    "method": "full_model",
                    "baseline_ns": m["baseline_ns"],
                    "optimized_ns": m["optimized_ns"],
                    "speedup": m["speedup"],
                    "num_ops": m.get("num_ops", 0),
                }
        else:
            checkpoints["model_715"] = {
                "method": "full_model",
                "baseline_ns": 0, "optimized_ns": 0,
                "speedup": None, "error": "no_data",
            }

        # Block v10 checkpoints
        if model in v10:
            vd = v10[model]
            if "error" in vd:
                for ckpt in CHECKPOINTS[1:]:
                    if ckpt not in checkpoints:
                        checkpoints[ckpt] = {
                            "method": "block_based",
                            "baseline_ns": 0, "optimized_ns": 0,
                            "speedup": None, "error": vd["error"],
                        }
            elif "checkpoints" in vd:
                for ckpt_name, cr in vd["checkpoints"].items():
                    entry["method"] = vd.get("method", "block_based")
                    entry["baseline_ns"] = vd.get("total_baseline_ns", 0)
                    checkpoints[ckpt_name] = {
                        "method": vd.get("method", "block_based"),
                        "baseline_ns": cr["baseline_ns"],
                        "optimized_ns": cr["optimized_ns"],
                        "speedup": cr["speedup"],
                        "blocks_used": cr.get("blocks_used", 0),
                        "errors": cr.get("errors", 0),
                    }

        comparison[model] = {
            "model": model,
            "method": entry["method"],
            "baseline_ns": entry["baseline_ns"],
            "checkpoints": checkpoints,
        }

    # Write JSON
    out_json = RESULTS_DIR / "checkpoint_comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Written → {out_json}")

    # Write CSV
    out_csv = RESULTS_DIR / "checkpoint_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        headers = ["Model", "Method", "Baseline(ns)"] + [f"Speedup_{c}" for c in CHECKPOINTS] + [f"Best_Checkpoint", "Best_Speedup"]
        writer = csv.writer(f)
        writer.writerow(headers)
        for model in ALL_MODELS:
            row = comparison.get(model, {})
            cps = row.get("checkpoints", {})
            best_sp = 0
            best_ck = ""
            speeds = []
            for ckpt in CHECKPOINTS:
                if ckpt in cps and cps[ckpt].get("speedup") is not None:
                    sp = cps[ckpt]["speedup"]
                    speeds.append(f"{sp:.4f}")
                    if sp > best_sp:
                        best_sp = sp
                        best_ck = ckpt
                else:
                    speeds.append("N/A")
            best_label = best_ck.replace("model_", "ckpt_") if best_ck else "N/A"
            best_str = f"{best_sp:.4f}x" if best_sp > 0 else "N/A"
            writer.writerow([
                model,
                row.get("method", ""),
                row.get("baseline_ns", 0),
                *speeds,
                best_label,
                best_str,
            ])
    print(f"Written → {out_csv}")

    # Summary table
    print(f"\n{'Model':<25} {'715':>10} {'791':>10} {'1500':>10} {'1983':>10} {'Best':>10}")
    print("-" * 75)
    for model in ALL_MODELS:
        row = comparison.get(model, {})
        cps = row.get("checkpoints", {})
        vals = []
        best = 0
        for ckpt in CHECKPOINTS:
            if ckpt in cps and cps[ckpt].get("speedup") is not None:
                sp = cps[ckpt]["speedup"]
                vals.append(f"{sp:.4f}")
                best = max(best, sp)
            else:
                vals.append("N/A")
        best_str = f"{best:.4f}" if best > 0 else "N/A"
        print(f"{model:<25} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10} {best_str:>10}")


if __name__ == "__main__":
    main()
