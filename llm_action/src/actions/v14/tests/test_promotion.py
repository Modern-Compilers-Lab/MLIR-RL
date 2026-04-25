from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 32, 32],
    },
    KernelType.CONV2D: {
        "tile_sizes": [1, 32, 1, 7, 8, 1, 1],
    },
}

if __name__ == "__main__":
    test_action(Promotion, params_per_kernel)
