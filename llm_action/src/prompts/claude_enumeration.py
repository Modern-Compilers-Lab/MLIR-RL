import argparse

from llm_action.src.models import KernelType
from llm_action.src.prompts.representation import get_training_code_templates_representation

def get_claude_run_prompt(kernel_type: KernelType, kernel_number: int) -> str:
    return f"""
INSTRUCTIONS: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/resources/prompts/v1/action_enumeration.md`

OUTPUT: Your output should be included in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/enumeration/`. You write the the next inexitent version creating its directory. Which means you:
- Lookup the latest version in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/enumeration/` and create a new directory with the next version number.
- Write the action enumeration to a file named `action_enumeration.json` in the new directory and your reasoning to a file named `reasoning.md` in the same directory. IMPORTANT: do not touch v0/ directory as it is reserved for example format reference.

CONTEXT BOUNDARIES: Every version must be independent of previous versions, the only reference you must consult is v0 only! Do not read any other files located in previous versions! Do not expect to find a content in the v0 enumeration, it is intentionally left empty to show you the file structure only.

INPUT: The RL System input will always be a single operation. Here are samples of the input operation (in MLIR format):
{get_training_code_templates_representation(include_instances=True, kernel_type=kernel_type, kernel_number=kernel_number)}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-type", type=KernelType, choices=list(KernelType), default=KernelType.MATMUL)
    parser.add_argument("--kernel-number", type=int, default=1)
    args = parser.parse_args()

    prompt = get_claude_run_prompt(kernel_type=args.kernel_type, kernel_number=args.kernel_number)
    print(prompt)
