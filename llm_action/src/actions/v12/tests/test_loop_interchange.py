from llm_action.src.models import KernelType
from llm_action.src.actions.v12.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "permutation": [1, 2, 0]
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_kernel)
