from llm_action.src.models import KernelType
from llm_action.src.actions.v7.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

# Note: conv2d is excluded because transform.structured.vectorize cannot handle
# the non-trivial affine indexing maps of convolution ops (window sliding).
# The action handles this gracefully by returning the original code.
params_per_kernel = {
    KernelType.MATMUL: {
        "vector_sizes": [4, 4, 4],
    },
    KernelType.GENERIC: {
        "vector_sizes": [2, 2, 2, 2, 4],
    },
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_kernel)
