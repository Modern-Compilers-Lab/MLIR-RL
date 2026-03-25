from llm_action.src.models import KernelType
from llm_action.src.actions.v6.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

# matmul has 3 iterators (M, N, K) -> permute to (N, M, K)
# conv2d has 7 iterators -> permute first two
# generic has 5 iterators -> permute first two
params_per_kernel = {
    KernelType.MATMUL: {
        "permutation": [1, 0, 2],
    },
    KernelType.CONV2D: {
        "permutation": [1, 0, 2, 3, 4, 5, 6],
    },
    KernelType.GENERIC: {
        "permutation": [1, 0, 2, 3, 4],
    },
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_kernel)
