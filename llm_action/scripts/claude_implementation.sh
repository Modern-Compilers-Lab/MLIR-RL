#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J claude_optim
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --qos=c2
#SBATCH --exclusive
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 7-00
#SBATCH -o llm_action/logs/jobs/claude_implementation_%j.out
#SBATCH -e llm_action/logs/jobs/claude_implementation_%j.err
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
# sbatch llm_action/scripts/claude_implementation.sh
# claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_implementation.py)"