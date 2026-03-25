from llm_action.src.models import KernelType
from llm_action.src.actions.v9.implementation.packing import Packing

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "packed_sizes": [32, 32, 32]
    }
}

if __name__ == "__main__":
    test_action(Packing, params_per_kernel)
