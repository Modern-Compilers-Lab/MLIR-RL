from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.peeling import Peeling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [48, 48, 48],
    },
    KernelType.CONV2D: {
        "tile_sizes": [1, 24, 1, 1, 24, 1, 1],
    },
}

if __name__ == "__main__":
    test_action(Peeling, params_per_kernel)
