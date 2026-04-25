from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.loop_unrolling import LoopUnrolling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [4, 4, 4],
        "unroll_factor": 2,
    },
    KernelType.CONV2D: {
        "tile_sizes": [1, 4, 1, 1, 4, 1, 1],
        "unroll_factor": 2,
    },
}

if __name__ == "__main__":
    test_action(LoopUnrolling, params_per_kernel)
