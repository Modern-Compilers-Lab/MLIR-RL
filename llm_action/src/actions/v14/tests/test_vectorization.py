from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "vector_sizes": [4, 4, 4],
    },
    KernelType.CONV2D: {
        "vector_sizes": [1, 4, 1, 1, 4, 1, 1],
    },
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_kernel)
