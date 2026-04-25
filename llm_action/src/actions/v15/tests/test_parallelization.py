from llm_action.src.models import KernelType
from llm_action.src.actions.v15.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "num_threads": 4,
    },
    KernelType.CONV2D: {
        "num_threads": 4,
    },
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_kernel)
