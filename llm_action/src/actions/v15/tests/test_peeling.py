from llm_action.src.models import KernelType
from llm_action.src.actions.v15.implementation.peeling import Peeling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 32, 0],
    },
}

if __name__ == "__main__":
    test_action(Peeling, params_per_kernel)
