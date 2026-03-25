from llm_action.src.models import KernelType
from llm_action.src.actions.v6.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

# Parallelize outer loops only (non-reduction dimensions)
# matmul: first dim (M) with 4 threads
# conv2d: first dim (N) with 4 threads
# generic: first dim with 4 threads
params_per_kernel = {
    KernelType.MATMUL: {
        "num_threads": [4, 0, 0],
    },
    KernelType.CONV2D: {
        "num_threads": [4, 0, 0, 0, 0, 0, 0],
    },
    KernelType.GENERIC: {
        "num_threads": [4, 0, 0, 0, 0],
    },
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_kernel)
