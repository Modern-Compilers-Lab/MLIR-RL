#!/bin/bash

# Define the resource requirements here using #SBATCH

#SBATCH -J claude_optim
#SBATCH -p compute
#SBATCH --reservation=c2
#SBATCH --exclusive
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 7-00
#SBATCH -o llm_action/logs/jobs/claude/enum_%j.out
#SBATCH -e llm_action/logs/jobs/claude/enum_%j.err
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
# claude /mcp

# Execute claude for once
claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_enumeration.py $KERNEL_ARGS)"

# Example usage:

# Matmul
## python llm_action/src/prompts/action_enumeration.py --intents_num_min 2 --intents_num_max 3 --transformations_num_min 2 --transformations_num_max 3 & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_enumeration.py --benchmark dataset_matmul)"

# Conv
## python llm_action/src/prompts/action_enumeration.py --intents_num_min 2 --intents_num_max 3 --transformations_num_min 2 --transformations_num_max 4 & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_enumeration.py --benchmark dataset_conv2d)"
## python llm_action/src/prompts/action_enumeration.py --intents_num_min 2 --intents_num_max 3 --transformations_num_min 2 --transformations_num_max 4 & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_enumeration.py --benchmark dataset_conv2d_img2col)"

# Pooling
## python llm_action/src/prompts/action_enumeration.py --intents_num_min 2 --intents_num_max 3 --transformations_num_min 2 --transformations_num_max 3 & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_enumeration.py --benchmark dataset_pooling)"

# Add
## python llm_action/src/prompts/action_enumeration.py --intents_num_min 2 --intents_num_max 3 --transformations_num_min 2 --transformations_num_max 3 & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_enumeration.py --benchmark dataset_add)"

# ReLu
## python llm_action/src/prompts/action_enumeration.py --intents_num_min 2 --intents_num_max 3 --transformations_num_min 2 --transformations_num_max 3 & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_enumeration.py --benchmark dataset_relu)"