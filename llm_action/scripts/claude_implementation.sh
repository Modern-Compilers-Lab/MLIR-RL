#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J claude_optim
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 7-00
#SBATCH -o llm_action/logs/jobs/claude/implem_%j.out
#SBATCH -e llm_action/logs/jobs/claude/implem_%j.err
#SBATCH --mail-user=kb5213@nyu.edu
#SBATCH --mail-type=ALL

# Resource requiremenmt commands end here

#Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate any environments if required
# conda activate llvm-build
conda activate mlir

# Parse arguments passed after sbatch: sbatch claude-enumeration.sh
KERNEL_ARGS="$@"

# Connect to the MCP server
claude /mcp

# Execute claude for once
claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py $KERNEL_ARGS)"

# Example usage:

# Matmul
## python llm_action/src/prompts/action_implementation.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py --benchmark dataset_matmul --limit 10)"

# Conv
## python llm_action/src/prompts/action_implementation.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py --benchmark dataset_conv2d --limit 10)"
## python llm_action/src/prompts/action_implementation.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py --benchmark dataset_conv2d_img2col)"

# Pooling
## python llm_action/src/prompts/action_implementation.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py --benchmark dataset_pooling --limit 10)"

# Add
## python llm_action/src/prompts/action_implementation.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py --benchmark dataset_add --limit 10)"

# ReLu
## python llm_action/src/prompts/action_implementation.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py --benchmark dataset_relu --limit 10)"

# ML
## python llm_action/src/prompts/action_implementation.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py --benchmark dataset_ml --limit 5)"