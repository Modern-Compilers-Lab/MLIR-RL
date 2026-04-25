from llm_action.src.models import KernelType
from llm_action.src.actions.v16.implementation.loop_unrolling import LoopUnrolling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {
        "tile_size": 32,
        "unroll_factor": 4,
    }
}

if __name__ == "__main__":
    test_action(LoopUnrolling, params_per_kernel)
