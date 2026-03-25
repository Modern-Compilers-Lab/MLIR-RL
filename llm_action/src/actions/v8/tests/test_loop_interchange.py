from llm_action.src.models import KernelType
from llm_action.src.actions.v8.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "permutation": [2, 0, 1],
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_kernel)
