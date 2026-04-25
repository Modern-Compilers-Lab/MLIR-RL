from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 32, 0],
    },
    KernelType.CONV2D: {
        "tile_sizes": [32, 32, 0, 0, 0, 0, 0],
    },
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_kernel)
