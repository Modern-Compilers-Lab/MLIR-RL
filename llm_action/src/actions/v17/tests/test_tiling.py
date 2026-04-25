from llm_action.src.models import KernelType
from llm_action.src.actions.v17.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {
        "tile_sizes": [32, 32]
    }
}

if __name__ == "__main__":
    test_action(Tiling, params_per_kernel)
