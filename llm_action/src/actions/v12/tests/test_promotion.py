from llm_action.src.models import KernelType
from llm_action.src.actions.v12.implementation.promotion import Promotion
from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "pad_multiple": 16
    }
}

if __name__ == "__main__":
    test_action(Promotion, params_per_kernel)
