from llm_action.src.models import KernelType
from llm_action.src.actions.v10.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "dimension": 2,
        "unroll_factor": 4,
    }
}

if __name__ == "__main__":
    test_action(LoopUnrolling, params_per_kernel)
