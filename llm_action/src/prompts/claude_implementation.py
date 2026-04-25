import argparse

from llm_action.src.models import KernelType
from llm_action.src.prompts.representation import get_training_code_templates_representation

def get_claude_run_prompt(kernel_type: KernelType, kernel_number: int) -> str:
    return f"""
INSTRUCTIONS: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/resources/prompts/v1/action_implementation.md`

REFERENCES:
- MCP Server Tools: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/MCP_REFERENCE.md`

OUTPUT: Your output should be included in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/`. Which means you:
- Lookup the latest version in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/` and create a `implementation/` and `tests/` subdirectories.
- Read the action enumeration in `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v<x>/enumeration/action_enumeration.json`
- Read the `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v0/implementation/name.py` and `/scratch/kb5213/workspace/MLIR-RL/llm_action/src/actions/v0/tests/name.py` for a reference on how to implement the actions and their respective unit tests.
- For every action in the action enumeration, you implement it and test it. If you need to execute the unit test for your implemented action, use `mlir` conda environment.
- Use a clean representative name for the action implementation and its test file. Example (tiling.py, test_tiling.py)
- Ensure that every `v0/tests/name.py` passes for the provided kernels, otherwise iterate and adjust the implementation. Ensure that the test files operate on the same kernel type and number as the input operation provided to you.
- Include all action is the `llm_action/src/actions/v<x>/registry.py` following the format of the `ACTION_CLASSES` list.
- Include all action tools in the `llm_action/src/actions/v<x>/mcp.py` following the format of the `name_tool` example.
- Add the actions MCP to `.mcp.json` under the key name `rl-action-v<x>` and ensure the command points to `llm_action.src.actions.v<x>.mcp` using the `mlir` conda environment.

CONTEXT BOUNDARIES: Every version must be independent of previous versions, the only reference you must consult is v0 only! Do not read any other files located in previous versions!

TESTING REQUIREMENTS: Your test implementation should follow the format of `v0/tests/name.py` strictly! 1) The action must run standalone, 2) Never leave a test without an execution (requiring extended preprocessing, that must be handled in the action definition). Just respect the test file code structure.

INPUT: The RL System input will always be a single operation. Here are samples of the input operation (in MLIR format):
Kernel parameters: --kernel-type={kernel_type} --kernel-number={kernel_number}
{get_training_code_templates_representation(include_instances=True, kernel_type=kernel_type, kernel_number=kernel_number)}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-type", type=KernelType, choices=list(KernelType), default=KernelType.MATMUL)
    parser.add_argument("--kernel-number", type=int, default=1)
    args = parser.parse_args()

    prompt = get_claude_run_prompt(kernel_type=args.kernel_type, kernel_number=args.kernel_number)
    print(prompt)
