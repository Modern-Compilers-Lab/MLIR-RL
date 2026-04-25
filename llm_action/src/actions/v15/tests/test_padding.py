from llm_action.src.models import KernelType
from llm_action.src.actions.v15.implementation.padding import Padding

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "pad_to_multiple_of": [8, 8, 8],
    },
}

if __name__ == "__main__":
    test_action(Padding, params_per_kernel)
