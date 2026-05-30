import argparse

from llm_action.src.config import PROJECT_ROOT
from llm_action.src.data.benchmarks import format_for_prompt

def get_claude_run_prompt(benchmark: str, limit: int = 10) -> str:
    return f"""
INSTRUCTIONS: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/resources/prompts/v1/action_implementation.md`

REFERENCES:
- MCP Server Tools: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/MCP.md`

MEMORY: A persistent scratchpad lives at `/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/memory/MEMORY.md`. Read it before you start so you can avoid re-discovering known pitfalls. As you work, append a short bullet whenever you hit a non-obvious bug, MLIR/transform-dialect quirk, pass-pipeline ordering issue, or test/CI gotcha — pair each bullet with the concrete fix. Keep entries one-line and scannable (`- bug: <symptom> -> fix: <hotfix>`); do not log routine progress, generic advice, or anything not bug-and-fix shaped. The goal is that future runs of this prompt suffer less from issues earlier runs already solved.

OUTPUT: Your output should be included in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/`. Which means you:
- Lookup the latest version in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/` and create a `implementation/` and `tests/` subdirectories.
- Read the action enumeration in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/enumeration/action_enumeration.json`
- Read the `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v0/implementation/name.py` and `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v0/tests/name.py` for a reference on how to implement the actions and their respective unit tests.
- For every action in the action enumeration, you implement it and test it. If you need to execute the unit test for your implemented action, use `mlir` conda environment.
- Use a clean representative name for the action implementation and its test file. Example (tiling.py, test_tiling.py)
- Ensure that every `v0/tests/name.py` passes for the provided kernels, otherwise iterate and adjust the implementation. Ensure that the test files operate on the same kernel type and number as the input operation provided to you.
- Include all action is the `llm_action/src/actions/v<x>/registry.py` following the format of the `ACTION_CLASSES` list.
- Include all action tools in the `llm_action/src/actions/v<x>/mcp.py` following the format of the `name_tool` example.
- Add the actions MCP to `.mcp.json` under the key name `rl-action-v<x>` and ensure the command points to `llm_action.src.actions.v<x>.mcp` using the `mlir` conda environment. Additionally, change `"llm_action.src.mcp.mcp_server"` to `"llm_action.src.mcp.mcp_server_minimal"`.

FILE WRITING: The directory creation won't work because of the spack error. Create the files directly using `Write`, which will create the directory structure.

CONTEXT BOUNDARIES: Every version must be independent of previous versions, the only reference you must consult is v0 only! Do not read any other files located in previous versions! Do not read other versions implementations! Recall importantly: do not read other versions content so you remain unbiased!

CONTEXT MANAGEMENT: Since this is a large task which you will probably compact conversation, you must reread `llm_action/src/prompts/claude_implementation.py` when this happens to stay aligned.

TESTING REQUIREMENTS: Your test implementation should follow the format of `v0/tests/name.py` strictly! 1) The action must run standalone, 2) Never leave a test without an execution (requiring extended preprocessing, that must be handled in the action definition). Just respect the test file code structure.

INPUT: The RL System input will always be a single operation. Below is the benchmark set you should reason about: per op family, the template, one concrete instance (full code), and the names/shapes of the other instances in the same family. The benchmarks are located in `{PROJECT_ROOT}/llm_action/data/benchmarks/{benchmark}/train/`.

{format_for_prompt(benchmark, split="train", limit=limit)}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Layer 2 action implementation prompt for Claude Code")
    parser.add_argument("--benchmark", type=str, default="standard",
                        help="Benchmark set under data/benchmarks/ (default: standard)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Max 'other shapes' listed per family in the embedded benchmark representation (token budget knob; default: 10)")
    args = parser.parse_args()

    prompt = get_claude_run_prompt(benchmark=args.benchmark, limit=args.limit)
    print(prompt)
