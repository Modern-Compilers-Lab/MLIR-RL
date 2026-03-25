from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.peeling import Peeling

from llm_action.src.actions.test import test_action

# Peel requires non-divisible tile_size to create a remainder loop
# MATMUL: M=128, tile by 30 -> remainder of 128%30=8
# CONV2D: N=128, tile by 30 -> remainder of 128%30=8
# GENERIC: dim0=8, tile by 3 -> remainder of 8%3=2
params_per_kernel = {
    KernelType.MATMUL: {
        "tile_size": 30,
        "dimension": 0,
    },
}

if __name__ == "__main__":
    test_action(Peeling, params_per_kernel)
