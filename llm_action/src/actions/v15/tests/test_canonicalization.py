from llm_action.src.models import KernelType
from llm_action.src.actions.v15.implementation.canonicalization import Canonicalization

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "apply_cse": 1,
    },
}

if __name__ == "__main__":
    test_action(Canonicalization, params_per_kernel)
