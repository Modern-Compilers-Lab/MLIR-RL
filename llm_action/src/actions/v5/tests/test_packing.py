from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.packing import Packing

from llm_action.src.actions.test import test_action

# MATMUL: 3 iteration dims (M=128, N=128, K=256)
# CONV2D: 7 iteration dims
# GENERIC: 5 iteration dims (8,8,16,8,32)
params_per_kernel = {
    KernelType.MATMUL: {
        "packed_sizes": [32, 32, 32],
    },
}

if __name__ == "__main__":
    test_action(Packing, params_per_kernel)
