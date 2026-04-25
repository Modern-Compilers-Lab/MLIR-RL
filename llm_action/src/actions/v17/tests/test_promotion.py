from llm_action.src.models import KernelType
from llm_action.src.actions.v17.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {
        "operands_to_promote": [0, 1]
    }
}

if __name__ == "__main__":
    test_action(Promotion, params_per_kernel)
