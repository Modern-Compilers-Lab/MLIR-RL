from llm_action.src.models import KernelType
from llm_action.src.actions.v16.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {
        "tile_sizes": [4, 32, 0],
    }
}

if __name__ == "__main__":
    test_action(Promotion, params_per_kernel)
