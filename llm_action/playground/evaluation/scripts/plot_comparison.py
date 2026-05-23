import csv
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from llm_action.src.config import EVALUATION_RESULTS_DIR, DATA_DIR

AUTO = EVALUATION_RESULTS_DIR
OLD_CSV = EVALUATION_RESULTS_DIR.parent / "mlir_rl" / "per_kernel.csv"
OUT_DIR = Path(__file__).resolve().parents[1] / "plots"

NEW_COLOR, OLD_COLOR, TORCH_COLOR = "#1a73e8", "#9aa0a6", "#f9ab00"
STATS = ["min", "q25", "median", "q75", "max"]
SHORT = {"matmul": "matmul", "conv_2d_nchw_fchw": "conv2d", "pooling_nchw_max": "pooling",
         "add": "add", "relu": "relu"}
CAT_TO_DATASET = {"matmul": "dataset_matmul", "conv_2d_nchw_fchw": "dataset_conv2d",
                  "add": "dataset_add", "relu": "dataset_relu", "pooling_nchw_max": "dataset_pooling"}
DATASET_ORDER = ["matmul", "conv_2d_nchw_fchw", "pooling_nchw_max", "add", "relu"]
SYSTEMS = [("New RL agent", NEW_COLOR, "new"), ("Old MLIR-RL", OLD_COLOR, "old"),
           ("PyTorch", TORCH_COLOR, "torch")]

# per-metric config. `torch` derives PyTorch's per-kernel value from (mlir, torch)
# baselines; `ref`/`views` drive the aggregated figures.
METRICS = {
    "speedup_over_mlir": dict(col="speedup", label="Speedup over MLIR baseline", unit="x",
                              kind="ratio", ref="mlir", higher_better=True, log=False,
                              parity=None, torch=lambda mlir, t: mlir / t,
                              views=["geomean"]),
    "speedup_vs_torch": dict(col="speedup_to_torch", label="Speedup over PyTorch", unit="x",
                             kind="ratio", ref="torch", higher_better=True, log=False,
                             parity=1.0, torch=lambda mlir, t: 1.0,
                             views=["geomean"]),
    "exec_time": dict(col="exec_time_ms", label="Execution time", unit="ms",
                      kind="time", ref=None, higher_better=False, log=True,
                      parity=None, torch=lambda mlir, t: t, views=["mean"]),
}


def geomean(v):
    return float(np.exp(np.mean(np.log([max(x, 1e-9) for x in v]))))


def fmt_val(v):
    return f"{v:.2f}" if v < 10 else f"{v:.0f}"


def shape_label(kernel, category):
    return kernel[len(category) + 1:].replace("_", "x")


def load_metric(path, col):
    """per_kernel.csv -> ({category: {kernel: {stat: value}}}, has_stats).

    has_stats True for the 5-number schema (`<col>_median` present); else the
    single `<col>` value is stored under "median".
    """
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        has_stats = f"{col}_median" in (reader.fieldnames or [])
        data = {}
        for row in reader:
            d = {s: float(row[f"{col}_{s}"]) for s in STATS} if has_stats \
                else {"median": float(row[col])}
            data.setdefault(row["category"], {})[row["kernel"]] = d
    return data, has_stats


def baselines(category):
    return json.loads((DATA_DIR / CAT_TO_DATASET[category] / "baselines.json").read_text())["eval"]


def _spread(ax, pos, st):
    """Gentle two-tier horizontal whisker: thin min..max, thick IQR, over the bar."""
    ax.errorbar(st["median"], pos, xerr=[[st["median"] - st["min"]], [st["max"] - st["median"]]],
                fmt="none", ecolor="black", elinewidth=0.7, alpha=0.35, capsize=2, zorder=5)
    ax.errorbar(st["median"], pos, xerr=[[st["median"] - st["q25"]], [st["q75"] - st["median"]]],
                fmt="none", ecolor="black", elinewidth=2.3, alpha=0.55, zorder=6)


def per_dataset_figure(category, key, cfg, sysdata, ent, show_spread):
    kernels = sorted(sysdata["new"], key=lambda k: sysdata["new"][k]["median"],
                     reverse=not cfg["higher_better"])  # best on top
    y = np.arange(len(kernels))
    h, log, u = 0.26, cfg["log"], cfg["unit"]
    allv = [d[k]["median"] for d in sysdata.values() for k in kernels]
    floor = min(v for v in allv if v > 0) * 0.6 if log else 0.0

    fig, ax = plt.subplots(figsize=(11, max(5, 0.62 * len(kernels) + 1.6)))
    legend = []
    for (name, color, sk), off in zip(SYSTEMS, (h, 0.0, -h)):
        data = sysdata[sk]
        vals = np.array([data[k]["median"] for k in kernels])
        ax.barh(y + off, vals - floor, height=h, left=floor, color=color, zorder=3)
        if show_spread and sk != "torch" and "max" in data[kernels[0]]:
            for k, yi in zip(kernels, y + off):
                _spread(ax, yi, data[k])
        ent_s = f" (ent {ent:g})" if sk == "new" else ""
        if cfg["kind"] == "ratio":
            lab = f"{name}{ent_s} — geomean {fmt_val(geomean(vals))}{u}"
        else:
            lab = f"{name}{ent_s} — mean {fmt_val(float(np.mean(vals)))}{u} · geomean {fmt_val(geomean(vals))}{u}"
        legend.append(Patch(facecolor=color, label=lab))

    if cfg["parity"] is not None:
        ax.axvline(cfg["parity"], color="black", lw=1.1, alpha=0.8)
        ax.text(cfg["parity"], len(kernels) - 0.4, " PyTorch parity", fontsize=8, va="top")
    if log:
        ax.set_xscale("log")
        ax.set_xlim(floor, max(allv) * 1.4)
    else:
        ax.set_xlim(0, max(allv) * 1.1)
    ax.set_yticks(y)
    ax.set_yticklabels([shape_label(k, category) for k in kernels],
                       fontsize=8 if len(kernels) <= 14 else 7)
    better = "higher is better" if cfg["higher_better"] else "lower is better, log scale"
    ax.set_xlabel(f"{cfg['label']}  ({u}, {better})", fontsize=10)
    ax.set_ylabel(f"{category} kernel", fontsize=9)
    ax.grid(axis="x", ls=":", alpha=0.4)
    ax.margins(y=0.01)
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.95)
    sub = ("bar = median, thick whisker = IQR, thin = min/max (9 runs)" if show_spread
           else "bar = single value (best training-logs eval); old = median")
    ax.set_title(f"{SHORT.get(category, category)}: Old MLIR-RL vs New RL agent vs PyTorch\n"
                 f"{cfg['label'].lower()} — {sub}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"{SHORT.get(category, category)}_{key}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}  ({len(kernels)} kernels, spread={show_spread})")


def _system_value(cfg, view, bundle_c, sk):
    """Aggregated value of a metric for one system in one dataset."""
    opt = bundle_c["opt"][sk]
    if cfg["kind"] == "time":
        return float(np.mean(opt))
    ref = bundle_c[cfg["ref"]]
    return geomean(ref / opt)


def aggregated_figure(key, cfg, view, bundle):
    cats = [c for c in DATASET_ORDER if c in bundle]
    x = np.arange(len(cats))
    w, log = 0.26, cfg["log"]
    vals = {sk: [_system_value(cfg, view, bundle[c], sk) for c in cats] for _, _, sk in SYSTEMS}
    flat = [v for s in vals.values() for v in s]
    floor = min(v for v in flat if v > 0) * 0.6 if log else 0.0

    fig, ax = plt.subplots(figsize=(max(7, 2.2 * len(cats) + 1), 5.5))
    for (name, color, sk), off in zip(SYSTEMS, (w, 0.0, -w)):
        heights = np.array(vals[sk])
        bars = ax.bar(x + off, heights - floor, width=w, bottom=floor, color=color, label=name, zorder=3)
        ax.bar_label(bars, labels=[fmt_val(v) for v in heights], padding=2, fontsize=7)

    if log:
        ax.set_yscale("log")
        ax.set_ylim(floor, max(flat) * 1.5)
    if cfg["parity"] is not None:
        ax.axhline(cfg["parity"], color="black", lw=1.0, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT.get(c, c) for c in cats], fontsize=10)
    better = "higher is better" if cfg["higher_better"] else "lower is better, log scale"
    view_label = {"geomean": "geomean", "mean": "mean"}[view]
    ax.set_ylabel(f"{cfg['label']}  ({cfg['unit']}, {better})", fontsize=10)
    ax.set_xlabel("dataset", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(f"Aggregated [{view_label}]: Old MLIR-RL vs New RL agent vs PyTorch\n"
                 f"{view_label} over kernels of {cfg['label'].lower()}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = OUT_DIR / f"aggregated_{key}_{view}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}  ({len(cats)} datasets)")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {}  # bundle[cat] = {"mlir","torch": arrays, "opt": {new,old,torch: arrays}}

    for d in sorted(glob.glob(str(AUTO / "*"))):
        pk = Path(d) / "per_kernel.csv"
        if not pk.exists():
            continue
        category = next(iter(load_metric(pk, "speedup")[0]))
        ent = json.loads((Path(d) / "results.json").read_text())["train_config"].get("ent_coef")
        bl = baselines(category)

        # ── per-dataset figures (one per metric) ──
        for key, cfg in METRICS.items():
            new, new_stats = load_metric(pk, cfg["col"])
            old, _ = load_metric(OLD_CSV, cfg["col"])
            if category not in new or category not in old:
                continue
            kernels = sorted(set(new[category]) & set(old[category]))
            torch = {k: {"median": cfg["torch"](bl[k]["mlir"], bl[k]["torch"])} for k in kernels}
            sysdata = {"new": new[category], "old": old[category], "torch": torch}
            per_dataset_figure(category, key, cfg, sysdata, ent, show_spread=new_stats)

        # ── aggregation bundle: per-kernel times (median opt for new/old) ──
        new_exec, _ = load_metric(pk, "exec_time_ms")
        old_exec, _ = load_metric(OLD_CSV, "exec_time_ms")
        kernels = sorted(set(new_exec[category]) & set(old_exec[category]))
        arr = lambda f: np.array([f(k) for k in kernels])
        bundle[category] = {
            "mlir": arr(lambda k: bl[k]["mlir"]),
            "torch": arr(lambda k: bl[k]["torch"]),
            "opt": {"new": arr(lambda k: new_exec[category][k]["median"]),
                    "old": arr(lambda k: old_exec[category][k]["median"]),
                    "torch": arr(lambda k: bl[k]["torch"])},
        }

    for key, cfg in METRICS.items():
        for view in cfg["views"]:
            if bundle:
                aggregated_figure(key, cfg, view, bundle)


if __name__ == "__main__":
    main()
