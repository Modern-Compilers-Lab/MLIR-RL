from llm_action.src.models import KernelType
from llm_action.src.actions.v13.implementation.padding import Padding

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "pad_multiple": 32
    }
}

if __name__ == "__main__":
    test_action(Padding, params_per_kernel)
