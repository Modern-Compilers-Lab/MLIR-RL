from llm_action.src.models import KernelType
from llm_action.src.actions.v13.implementation.packing import Packing

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "packed_sizes": [8, 16, 4]
    }
}

if __name__ == "__main__":
    test_action(Packing, params_per_kernel)
