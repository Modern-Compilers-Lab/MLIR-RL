#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J claude_explore
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 7-00
#SBATCH -o llm_action/logs/jobs/claude/explore_%j.out
#SBATCH -e llm_action/logs/jobs/claude/explore_%j.err
#SBATCH --mail-user=kb5213@nyu.edu
#SBATCH --mail-type=ALL

# Resource requirement commands end here

# Add the lines for running your code/application
module load miniconda-nobashrc 2> /dev/null
eval "$(conda shell.bash hook)"

# Activate environment
conda activate mlir

# Parse arguments passed after sbatch
KERNEL_ARGS="$@"

# Connect to the MCP server
claude /mcp

# Execute claude for schedule exploration
claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py $KERNEL_ARGS)"

# Example usage:

# Matmul
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v29 --benchmark dataset_matmul)"
# Conv
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v30 --benchmark dataset_conv2d)"