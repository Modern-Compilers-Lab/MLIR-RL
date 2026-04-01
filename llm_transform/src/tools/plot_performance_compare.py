import argparse
import re
import math
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

PARENT_DIR = Path(__file__).parents[2]


def parse_performance_log(log_path: Path):
    entries = defaultdict(list)
    pattern = re.compile(
        r"(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| id=(?P<id>\S+) \| speedup=(?P<speedup>[\d.]+)x"
    )
    with open(log_path) as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                code_id = m.group("id")
                speedup = float(m.group("speedup"))
                entries[code_id].append(speedup)
    return entries


def plot_compare(all_entries: dict[int, dict[str, list[float]]], output_path: Path):
    # Collect all code_ids across all experiments
    code_ids = sorted({cid for entries in all_entries.values() for cid in entries})

    ncols = min(3, len(code_ids))
    nrows = math.ceil(len(code_ids) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10 * ncols, 6 * nrows), squeeze=False)

    for idx, code_id in enumerate(code_ids):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        for exp_id, entries in sorted(all_entries.items()):
            if code_id in entries:
                speedups = entries[code_id]
                ax.plot(range(1, len(speedups) + 1), speedups, label=f"Exp {exp_id}")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Speedup (vs PyTorch)")
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="Baseline")
        ax.set_title(f"ID {code_id}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(code_ids), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle("Speedup Comparison Across Experiments", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare speedup across multiple experiments per CODE_ID")
    parser.add_argument("experiment_ids", type=int, nargs="+", help="Experiment IDs to compare")
    parser.add_argument("--stats-dir", default="logs/stats", help="Stats directory (default: logs/stats)")
    parser.add_argument("-o", "--output", default=None, help="Output file path (default: stats-dir/compare.png)")
    args = parser.parse_args()

    stats_dir = PARENT_DIR / Path(args.stats_dir)
    all_entries = {}

    for exp_id in args.experiment_ids:
        log_path = stats_dir / str(exp_id) / "performance.log"
        if not log_path.exists():
            print(f"Warning: {log_path} not found, skipping experiment {exp_id}")
            continue
        entries = parse_performance_log(log_path)
        if entries:
            all_entries[exp_id] = entries
        else:
            print(f"Warning: No valid entries in {log_path}, skipping experiment {exp_id}")

    if not all_entries:
        print("Error: No data found for any experiment")
        return 1

    output_path = Path(args.output) if args.output else stats_dir / "compare.png"
    plot_compare(all_entries, output_path)
    return 0


if __name__ == "__main__":
    exit(main())
