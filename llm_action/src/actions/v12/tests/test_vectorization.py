from llm_action.src.models import KernelType
from llm_action.src.actions.v12.implementation.vectorization import Vectorization
from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "vector_sizes": [4, 4, 4]
    }
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_kernel)
