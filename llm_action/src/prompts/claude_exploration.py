import argparse

from llm_action.src.config import PROJECT_ROOT
from llm_action.src.data.benchmarks import format_for_prompt


def get_claude_run_prompt(action_version: str, benchmark: str, limit: int = 10) -> str:
    return f"""
INSTRUCTIONS: Available in `{PROJECT_ROOT}/llm_action/resources/prompts/v1/schedule_exploration.md`

REFERENCES:
- MCP Servers Tools: `mlir-tools` (`/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/MCP_MINIMAL.md`) and `rl-action-v{action_version}`.

LOGS: Write your exploration log to `{PROJECT_ROOT}/llm_action/logs/mcp/{action_version}/<kernel_name>_<datetime(YYMMDDHHMM)>.md`
- Create the directory if it does not exist.
- Write incrementally after each phase to preserve progress.

FILE WRITING: The directory creation won't work because of the spack error. Create the files directly using `Write`, which will create the directory structure.

BENCHMARKS:
{format_for_prompt(benchmark, split="train", annotate_baselines=True, limit=limit)}

TASK: Explore per kernel family, using a token-efficient kernel budget per phase:
- **Phases 1-2 (single actions + pairwise composability):** use just **1 representative kernel per family** — these results are structural and do not vary meaningfully across shapes, so one kernel is sufficient.
- **Phases 3-4 (multi-step schedule shapes + selection across cases):** use **3 kernels per family** representing distinct patterns (e.g. small / medium / large, or any partition you judge relevant), so you can observe which schedule shape wins for which subset.

Steps:
1. Read the kernel MLIR file(s) you need for the current phase from `{PROJECT_ROOT}/llm_action/data/benchmarks/{benchmark}/train/<kernel_name>.mlir`.
2. Establish baselines (MLIR base execution via execute_mlir_code) for each kernel you use.
3. Run the phased exploration from the instructions (single actions, pairwise compositions, multi-step schedule shapes) with the per-phase kernel budget above, using the action MCP tools.
4. Measure each candidate schedule SHAPE once with a reasonable valid probe parameter set (execute_mlir_code and measure_speedup) to rank shapes per kernel case — parameter optimization is the RL policy's job, not yours. Track the winning schedules shape per kernel subset, not just one global best.
5- In phase 3 and 4, and in order to optimize for the explored schedule and provide high quality schedules especially, run parallel agents to focus on each kernel family subset, so you can explore more candidates in the same overall time budget. Instruct the agents to comprehensively explore and optimize for efficient schedules to highly improve RL sample efficiency, keeping 3-5 ordered schedules that have high potential of surpassing PyTorch. You `the main agent` can do phase 1 and 2, then assign the phase 3 and 4 kernels to `parallel agents` with instructions to explore comprehensively and optimize for efficient schedules, then gather their findings to synthesize the final structures.
6. Write the structured exploration log following the format in the instructions. Make sure to write progressively so you don't lose track when conversation gets compacted.
7. After exploration, synthesize TWO structures and append BOTH to `llm_action/src/actions/{action_version}/registry.py` directly below the existing `ACTION_CLASSES` list:
   a. `SCHEDULE_GRAPH` (PRIMARY) — a per-family allowlist of high-value schedule paths (action-name skeletons, no parameters), per the "Schedule Graph Synthesis" section. Key by op family; for each family include the distinct winning skeletons across shape subsets (coverage/balance), but keep the set small and curated (parsimony — do not enumerate every composable sequence).
   b. `ACTION_DEPENDENCIES` (LEGACY) — the cross-kernel denylist per the "Dependency Graph Synthesis" section. Only include block edges that held on every kernel where both endpoints were applicable — kernel-specific quirks do not belong in the graph. Recall that the structure is if Action X (key) has executed, Actions Yi (values) become unavailable.
   The two structures must be consistent: never encode a SCHEDULE_GRAPH path that includes a blocked transition.

BUDGET: Up to "Unlimited (ensure full coverage)" candidates per kernel. Prioritize breadth in Phases 1-2; in Phases 3-4 prioritize breadth of distinct schedule SHAPES across cases, not parameter depth.

CRITICAL: Use ONLY the action MCP tools (rl-action-{action_version}) for transformations. Do NOT write raw Transform dialect code. Do NOT modify action implementations. Your job is to explore what the existing actions can achieve when composed.

EMPIRICAL RIGOR: Test every action pair in the composability matrix by actually calling the tools. NEVER assume an action is "terminal" or incompatible without testing it. NEVER write "N/A" — every cell must reflect an actual tool invocation. Previous explorations incorrectly assumed Promotion was terminal without testing; it was not.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Layer 3 schedule exploration prompt for Claude Code")
    parser.add_argument("--action-version", type=str,
                        help="Action version (e.g., v10)", default="v10")
    parser.add_argument("--benchmark", "--benchmarks-name", dest="benchmark", type=str, default="standard",
                        help="Benchmark set under data/benchmarks/ (default: standard)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max 'other shapes' listed per family in the embedded benchmark representation (token budget knob; default: 10)")
    args = parser.parse_args()

    prompt = get_claude_run_prompt(
        action_version=args.action_version,
        benchmark=args.benchmark,
        limit=args.limit,
    )
    print(prompt)
