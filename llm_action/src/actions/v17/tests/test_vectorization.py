from llm_action.src.models import KernelType
from llm_action.src.actions.v17.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {
        "vector_sizes": [4, 1, 4]
    }
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_kernel)
