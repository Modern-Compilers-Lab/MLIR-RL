from llm_action.src.models import KernelType
from llm_action.src.actions.v14.implementation.loop_distribution import LoopDistribution

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "dimension": 0,
        "chunk_size": 64,
    },
    KernelType.CONV2D: {
        "dimension": 1,
        "chunk_size": 128,
    },
}

if __name__ == "__main__":
    test_action(LoopDistribution, params_per_kernel)
