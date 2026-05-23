import argparse

from llm_action.src.config import PROJECT_ROOT
from llm_action.src.data.benchmarks import format_for_prompt


def get_claude_run_prompt(action_version: str, benchmark: str) -> str:
    return f"""
INSTRUCTIONS: Available in `{PROJECT_ROOT}/llm_action/resources/prompts/v1/schedule_exploration.md`

REFERENCES:
- MCP Servers Tools: `mlir-tools` (`/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/MCP_MINIMAL.md`) and `rl-action-v{action_version}`.

LOGS: Write your exploration log to `{PROJECT_ROOT}/llm_action/logs/mcp/{action_version}/<kernel_name>_<datetime(YYMMDDHHMM)>.md`
- Create the directory if it does not exist.
- Write incrementally after each phase to preserve progress.

FILE WRITING: The directory creation won't work because of the spack error. Create the files directly using `Write`, which will create the directory structure.

BENCHMARKS:
{format_for_prompt(benchmark, split="train", annotate_baselines=True)}

TASK: For 1 representative kernel per kernel type listed above:
1. Read the kernel MLIR file from `{PROJECT_ROOT}/llm_action/data/benchmarks/{benchmark}/train/<kernel_name>.mlir`.
2. Establish baselines (MLIR base execution via execute_mlir_code).
3. Systematically explore single actions, pairwise compositions, and multi-step schedules using the action MCP tools.
4. Measure performance of each successful schedule using execute_mlir_code and measure_speedup.
5. Write the structured exploration log following the format in the instructions. Make sure to write progressively so you don't lose track when conversation gets compacted.
6. After the chosen kernel is explored, synthesize a single cross-kernel `ACTION_DEPENDENCIES` dict per the "Dependency Graph Synthesis" section of the instructions, and append it to `llm_action/src/actions/{action_version}/registry.py` directly below the existing `ACTION_CLASSES` list. Only include block edges that held on every kernel where both endpoints were applicable — kernel-specific quirks do not belong in the graph.

BUDGET: Up to "Unlimited (ensure full coverage)" candidates per kernel. Prioritize breadth in Phases 1-2, depth in Phases 3-4.

CRITICAL: Use ONLY the action MCP tools (rl-action-{action_version}) for transformations. Do NOT write raw Transform dialect code. Do NOT modify action implementations. Your job is to explore what the existing actions can achieve when composed.

EMPIRICAL RIGOR: Test every action pair in the composability matrix by actually calling the tools. NEVER assume an action is "terminal" or incompatible without testing it. NEVER write "N/A" — every cell must reflect an actual tool invocation. Previous explorations incorrectly assumed Promotion was terminal without testing; it was not.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Layer 3 schedule exploration prompt for Claude Code")
    parser.add_argument("--action-version", type=str,
                        help="Action version (e.g., v10)", default="v10")
    parser.add_argument("--benchmark", "--benchmarks-name", dest="benchmark", type=str, default="standard",
                        help="Benchmark set under data/benchmarks/ (default: standard)")
    args = parser.parse_args()

    prompt = get_claude_run_prompt(
        action_version=args.action_version,
        benchmark=args.benchmark,
    )
    print(prompt)
