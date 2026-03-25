from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

# Vector sizes should be small enough to fit in AVX2 registers
# and satisfy the safety contract (N <= 1024)
# MATMUL: 3 dims -> small vector sizes for tile+vectorize pattern
# CONV2D: 7 dims -> very small to stay under limits
# GENERIC: 5 dims -> small sizes
params_per_kernel = {
    KernelType.MATMUL: {
        "vector_sizes": [4, 4, 4],
    },
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_kernel)
