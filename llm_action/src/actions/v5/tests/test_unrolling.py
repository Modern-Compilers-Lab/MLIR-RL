from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.unrolling import Unrolling

from llm_action.src.actions.test import test_action

# Tile on one dimension, then unroll the generated loop
# MATMUL: tile dim 0 (M=128) by 32, unroll by 2
# CONV2D: tile dim 0 (N=128) by 16, unroll by 2
# GENERIC: tile dim 0 (8) by 4, unroll by 2
params_per_kernel = {
    KernelType.MATMUL: {
        "tile_size": 32,
        "dimension": 0,
        "unroll_factor": 2,
    },
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_kernel)
