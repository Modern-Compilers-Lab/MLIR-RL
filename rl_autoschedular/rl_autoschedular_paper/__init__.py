"""Core module for MLIR RL.

This module sets up the computation device (CUDA or CPU).
"""

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == __package__:
    print("Using device:", device.type)
