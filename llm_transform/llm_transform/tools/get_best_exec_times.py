"""Read a `state.json` (best speedups per id) and look up the corresponding
`exec_time_ns` for each id from the matching `performance.log`.

Usage:
    python get_best_exec_times.py <experiment_dir>

Where <experiment_dir> contains both `best/state.json` and `performance.log`
(e.g. logs/stats/20).
"""

import argparse
import json
import re
import sys
from pathlib import Path


LINE_RE = re.compile(
    r"id=(?P<id>\S+)\s*\|\s*speedup=(?P<speedup>[\d.]+)x\s*\|\s*exec_time_ns=(?P<ns>\d+)"
)


def parse_log(log_path: Path):
    entries: dict[str, list[tuple[float, int]]] = {}
    with log_path.open() as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            speedup_str = m.group("speedup")
            if speedup_str.startswith("."):
                speedup_str = "0" + speedup_str
            speedup = float(speedup_str)
            entries.setdefault(m.group("id"), []).append(
                (speedup, int(m.group("ns")))
            )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment directory containing best/state.json and performance.log",
    )
    args = parser.parse_args()

    state_path = args.experiment_dir / "best" / "state.json"
    log_path = args.experiment_dir / "performance.log"

    if not state_path.is_file():
        print(f"error: {state_path} not found", file=sys.stderr)
        return 1
    if not log_path.is_file():
        print(f"error: {log_path} not found", file=sys.stderr)
        return 1

    state = json.loads(state_path.read_text())
    log_entries = parse_log(log_path)

    print(f"{'id':<16} {'speedup':>10} {'exec_time_ns':>16}")
    print("-" * 46)
    for code_id, best_speedup in state.items():
        candidates = log_entries.get(code_id, [])
        # Match the entry whose speedup is closest to the recorded best.
        match = min(
            candidates,
            key=lambda e: abs(e[0] - best_speedup),
            default=None,
        )
        if match is None:
            print(f"{code_id:<16} {best_speedup:>10.4f} {'<missing>':>16}")
        else:
            print(f"{code_id:<16} {best_speedup:>10.4f} {match[1]:>16}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
