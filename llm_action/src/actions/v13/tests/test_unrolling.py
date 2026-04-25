from llm_action.src.models import KernelType
from llm_action.src.actions.v13.implementation.unrolling import Unrolling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "unroll_factor": 4
    }
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_kernel)
