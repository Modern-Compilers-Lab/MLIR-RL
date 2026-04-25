from llm_action.src.models import KernelType
from llm_action.src.actions.v12.implementation.loop_distribution import LoopDistribution
from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "split_factor": 4
    }
}

if __name__ == "__main__":
    test_action(LoopDistribution, params_per_kernel)
