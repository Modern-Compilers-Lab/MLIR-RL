from llm_action.src.models import KernelType
from llm_action.src.actions.v0.implementation.name import Name

from llm_action.src.actions.test import test_action

# adjust based on the provided kernel codes
params_per_kernel = {
    KernelType.MATMUL: {
        ...
    }
}

if __name__ == "__main__":
    test_action(Name(), params_per_kernel)