import argparse

from llm_action.src.data.benchmarks import format_for_prompt

def get_claude_run_prompt(benchmark: str) -> str:
    return f"""
INSTRUCTIONS: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/resources/prompts/v1/action_enumeration.md`

OUTPUT: Your output should be included in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/enumeration/`. You write the the next inexitent version creating its directory. Which means you:
- Lookup the latest version in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/enumeration/` and create a new directory with the next version number.
- Write the action enumeration to a file named `action_enumeration.json` in the new directory and your reasoning to a file named `reasoning.md` in the same directory. IMPORTANT: do not touch v0/ directory as it is reserved for example format reference.

FILE WRITING: The directory creation won't work because of the spack error. Create the files directly using `Write`, which will create the directory structure.

CONTEXT BOUNDARIES: Every version must be independent of previous versions, the only reference you must consult is v0 only! Do not read any other files located in previous versions! Do not expect to find a content in the v0 enumeration, it is intentionally left empty to show you the file structure only.

INPUT: The RL System input will always be a single operation. Below is the benchmark set you should reason about: per op family, the template, one concrete instance (full code), and the names/shapes of the other instances in the same family.
{format_for_prompt(benchmark, split="train")}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Layer 1 action enumeration prompt for Claude Code")
    parser.add_argument("--benchmark", type=str, default="standard",
                        help="Benchmark set under data/benchmarks/ (default: standard)")
    args = parser.parse_args()

    prompt = get_claude_run_prompt(benchmark=args.benchmark)
    print(prompt)
