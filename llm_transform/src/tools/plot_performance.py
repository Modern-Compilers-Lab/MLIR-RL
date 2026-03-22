import argparse
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

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


def plot(entries: dict, output_path: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for code_id, speedups in sorted(entries.items()):
        ax.plot(range(1, len(speedups) + 1), speedups, label=f"ID {code_id}")
    ax.set_yscale("log")  # <--- Add this line
    # Optional: Format the ticks so they don't look like scientific notation
    from matplotlib.ticker import ScalarFormatter
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Speedup (vs PyTorch)")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, label="PyTorch baseline")
    ax.set_title("Speedup over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    # import numpy as np
    # all_values = [v for subs in entries.values() for v in subs]
    # if all_values:
    #     # Set the top of the graph to the 95th percentile + a little buffer
    #     ymax = np.percentile(all_values, 95) * 1.1
    #     ax.set_ylim(0, ymax)
    fig.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot speedup over time per CODE_ID")
    parser.add_argument("experiment_id", type=int, help="Experiment ID")
    parser.add_argument("--stats-dir", default="logs/stats", help="Stats directory (default: logs/stats)")
    args = parser.parse_args()

    experiment_dir = PARENT_DIR / Path(args.stats_dir) / str(args.experiment_id)
    log_path = experiment_dir / "performance.log"
    if not log_path.exists():
        print(f"Error: {log_path} not found")
        return 1

    entries = parse_performance_log(log_path)
    if not entries:
        print(f"Error: No valid entries found in {log_path}")
        return 1

    output_path = experiment_dir / "performance.png"
    plot(entries, output_path)
    return 0


if __name__ == "__main__":
    exit(main())
