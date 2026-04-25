from llm_action.src.models import KernelType
from llm_action.src.actions.v17.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {
        "permutation": [1, 0, 2, 3, 4, 5, 6]
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_kernel)
