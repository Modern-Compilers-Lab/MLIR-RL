from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "permutation": [1, 0, 2],
    },
    KernelType.CONV2D: {
        "permutation": [0, 2, 1, 3, 4, 5, 6],
    },
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_kernel)
