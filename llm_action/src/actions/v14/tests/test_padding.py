from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.padding import Padding

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "pad_to_multiple_of": [48, 48, 48],
    },
    KernelType.CONV2D: {
        "pad_to_multiple_of": [1, 48, 1, 1, 48, 1, 1],
    },
}

if __name__ == "__main__":
    test_action(Padding, params_per_kernel)
