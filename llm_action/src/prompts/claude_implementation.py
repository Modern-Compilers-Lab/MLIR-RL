import argparse

from llm_action.src.prompts.representation import get_training_code_templates_representation

def get_claude_run_prompt() -> str:
    return f"""
INSTRUCTIONS: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/resources/prompts/v1/action_implementation.md`

REFERENCES:
- MCP Server Tools: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/MCP_REFERENCE.md`

OUTPUT: Your output should be included in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/`. Which means you:
- Lookup the latest version in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/` and create a `implementation/` and `tests/` subdirectories.
- Read the action enumeration in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/enumeration/action_enumeration.json`
- Read the `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v0/implementation/action_1.py` and `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v0/tests/test_action_1.py` for a reference on how to implement the actions and their respective unit tests.
- For every action in the action enumeration, you implement it and test it. If you need to execute the unit test for your implemented action, use `mlir` conda environment.
- Use a clean representative name for the action implementation and its test file. Example (tiling.py, test_tiling.py)
- Ensure that every `test_action.py` passes for all kernel, otherwise iterate and adjust the implementation.

CONTEXT BOUNDARIES: Every version must be independent of previous versions, the only reference you must consult is v0 only! Do not read any other files located in previous versions!

INPUT: The RL System input will always be a single operation. Here are samples of the input operation (in MLIR format):
{get_training_code_templates_representation(include_instances=True)}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--kernel-type", type=KernelType, choices=list(KernelType), default=KernelType.MATMUL)
    args = parser.parse_args()

    prompt = get_claude_run_prompt()
    print(prompt)
