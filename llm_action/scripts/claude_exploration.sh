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
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v32 --benchmark dataset_matmul)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v40 --benchmark dataset_matmul)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v41 --benchmark dataset_matmul --limit 20)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v42 --benchmark dataset_matmul --limit 20)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v46 --benchmark dataset_matmul --limit 20)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v48 --benchmark dataset_matmul --limit 20)"

# Conv
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v33 --benchmark dataset_conv2d)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v43 --benchmark dataset_conv2d --limit 20)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v49 --benchmark dataset_conv2d --limit 20)"


## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v36 --benchmark dataset_conv2d_img2col)"

# Pooling
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v34 --benchmark dataset_pooling)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v50 --benchmark dataset_pooling --limit 20)"

# Add
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v35 --benchmark dataset_add)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v51 --benchmark dataset_add --limit 20)"


# ReLu
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v37 --benchmark dataset_relu)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v52 --benchmark dataset_relu --limit 20)"

# ML
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v39 --benchmark dataset_ml)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v45 --benchmark dataset_ml --limit 10)"
## python llm_action/src/prompts/schedule_exploration.py & claude --dangerously-skip-permissions "$(python llm_action/src/prompts/claude_exploration.py --action-version v53 --benchmark dataset_ml --limit 10)"