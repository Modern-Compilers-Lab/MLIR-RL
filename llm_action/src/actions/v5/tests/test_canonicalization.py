from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.canonicalization import Canonicalization

from llm_action.src.actions.test import test_action

# Canonicalization takes no parameters
params_per_kernel = {
    KernelType.MATMUL: {},
}

if __name__ == "__main__":
    test_action(Canonicalization, params_per_kernel)
