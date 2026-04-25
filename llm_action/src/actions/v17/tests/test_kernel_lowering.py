from llm_action.src.models import KernelType
from llm_action.src.actions.v17.implementation.kernel_lowering import KernelLowering

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.CONV2D: {}
}

if __name__ == "__main__":
    test_action(KernelLowering, params_per_kernel)
