import argparse
import json
import os

from llm_action.src.config import PROJECT_ROOT

def get_benchmarks_kernel_list(benchmarks_name: str) -> str:
    benchmarks_dir = PROJECT_ROOT / "llm_action" / "data" / "benchmarks" / benchmarks_name
    baselines_path = benchmarks_dir / "baselines.json"

    mlir_files = sorted(benchmarks_dir.glob("*.mlir"))
    if not mlir_files:
        return f"No .mlir files found in {benchmarks_dir}"

    baselines = {}
    if baselines_path.exists():
        try:
            baselines = json.loads(baselines_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    lines = [f"Benchmark kernels in {benchmarks_name}/:"]
    for f in mlir_files:
        kernel_name = f.stem
        baseline_info = f" (baseline: {baselines[kernel_name]:.2f} ms)" if kernel_name in baselines else ""
        lines.append(f"- {f}{baseline_info}")

    return "\n".join(lines)

def get_claude_run_prompt(action_version: str, benchmarks_name: str) -> str:
    return f"""
INSTRUCTIONS: Available in `{PROJECT_ROOT}/llm_action/resources/prompts/v1/schedule_exploration.md`

REFERENCES:
- MCP Servers Tools: `mlir-tools` and `rl-action-v{action_version}`.

LOGS: Write your exploration log to `{PROJECT_ROOT}/llm_action/logs/mcp/{action_version}/<kernel_name>_<datetime(YYMMDDHHMM)>.md`
- Create the directory if it does not exist.
- Write incrementally after each phase to preserve progress.

BENCHMARKS:
{get_benchmarks_kernel_list(benchmarks_name)}

TASK: For each kernel listed above:
1. Read the kernel MLIR file.
2. Establish baselines (MLIR base execution via execute_mlir_code).
3. Systematically explore single actions, pairwise compositions, and multi-step schedules using the action MCP tools.
4. Measure performance of each successful schedule using execute_mlir_code and measure_speedup.
5. Write the structured exploration log following the format in the instructions.

BUDGET: Up to "Unlimited (ensure full coverage)" candidates per kernel. Prioritize breadth in Phases 1-2, depth in Phases 3-4.

CRITICAL: Use ONLY the action MCP tools (rl-action-{action_version}) for transformations. Do NOT write raw Transform dialect code. Do NOT modify action implementations. Your job is to explore what the existing actions can achieve when composed.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Layer 3 schedule exploration prompt for Claude Code")
    parser.add_argument("--action-version", type=str,
                        help="Action version (e.g., v10)", default="v10")
    parser.add_argument("--benchmarks-name", type=str, default="matmul",
                        help="Benchmarks subdirectory name (e.g., matmul)")
    args = parser.parse_args()

    prompt = get_claude_run_prompt(
        action_version=args.action_version,
        benchmarks_name=args.benchmarks_name,
    )
    print(prompt)
