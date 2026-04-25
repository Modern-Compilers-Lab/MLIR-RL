from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.im2col_lowering import Im2colLowering

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "enable": 1,
    },
    KernelType.CONV2D: {
        "enable": 1,
    },
}

if __name__ == "__main__":
    test_action(Im2colLowering, params_per_kernel)
