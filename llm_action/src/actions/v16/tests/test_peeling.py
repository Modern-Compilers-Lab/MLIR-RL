from llm_action.src.models import KernelType
from llm_action.src.actions.v16.implementation.peeling import Peeling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {
        "tile_size": 16,
        "peel_front": 0,
    }
}

if __name__ == "__main__":
    test_action(Peeling, params_per_kernel)
