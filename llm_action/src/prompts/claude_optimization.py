import argparse

from llm_action.src.models import KernelType
from llm_action.src.utils.persistence import load_kernel_code

def get_claude_run_prompt(kernel_type: KernelType, kernel_number: int) -> str:
    return f"""
INSTRUCTIONS: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/resources/prompts/v1/optimization.md`

REFERENCES:
- MLIR Optimization: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/MLIR_OPTIMIZATION_REFERENCE.md`
- MCP Server Tools: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/docs/MCP_REFERENCE.md`

MEMORY: Available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/memory/MLIR_TECHNICAL_MEMORY.md`. You can append to it with any new information you learn during the optimization process (bugs, preprocessing steps, transformations, pass pipelines, etc). Keep it concise and organized for easy reference. Only use this memory file! Do not use any other sources of memory (DO NOT read from archive/).

PREVIOUS EXPERIMENTS: You can learn only from the previous experiments on the same kernel type, which are available in `/scratch/kb5213/workspace/MLIR-RL/llm_action/logs/claude/`. You can use these logs to understand the transformations applied, the pass pipelines used, and the performance improvements achieved in previous runs. Do not use any information from other codebase files or any other sources (DO NOT read from archive/).

LOGS: Output summarized human-readable details of your progress in real-time (append your progress once you test every candidate, to mitigate issues of limit and losing progress) to a log file (the directory is already created) `/scratch/kb5213/workspace/MLIR-RL/llm_action/logs/claude/optimization_<code_identifier(matmul_m_k_n)>_<datetime(yymmddhhmm)>.log`. Finally log the detailed transformation and pass pipeline for the best candidate on convergence or surpassing PyTorch performance.

BUDGET: Experiement up to 50 candidates (transformations + pass pipelines). You can apply multiple transformations and passes in a single candidate, but be mindful of the combinatorial explosion of the search space. Prioritize transformations and passes that are more likely to yield significant performance improvements based on your understanding and previous experiments.

CRITICAL: Focus on reading the mentioned references and only use the MCP tools for transformations, pass pipelines, and benchmarking. Do not use any other tools or resources. You have all you need in the provided references and MCP tools to succeed!

INPUT MLIR CODE:
{load_kernel_code(kernel_type, kernel_number)}
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-type", type=KernelType, choices=list(KernelType), default=KernelType.MATMUL)
    parser.add_argument("--kernel-number", type=int, default=1)
    args = parser.parse_args()

    prompt = get_claude_run_prompt(kernel_type=args.kernel_type, kernel_number=args.kernel_number)
    print(prompt)
