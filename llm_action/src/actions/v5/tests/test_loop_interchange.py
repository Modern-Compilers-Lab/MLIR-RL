from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

# MATMUL: 3 iteration dims -> permutation of [0,1,2]
# CONV2D: 7 iteration dims -> permutation of [0..6]
# GENERIC: 5 iteration dims -> permutation of [0..4]
params_per_kernel = {
    KernelType.MATMUL: {
        "permutation": [1, 0, 2],
    },
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_kernel)
