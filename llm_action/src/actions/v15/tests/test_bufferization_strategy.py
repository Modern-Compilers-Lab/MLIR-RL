from llm_action.src.models import KernelType
from llm_action.src.actions.v15.implementation.bufferization_strategy import BufferizationStrategy

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "strategy": 0,
    },
}

if __name__ == "__main__":
    test_action(BufferizationStrategy, params_per_kernel)
