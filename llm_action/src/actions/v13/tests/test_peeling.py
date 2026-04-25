from llm_action.src.models import KernelType
from llm_action.src.actions.v13.implementation.peeling import Peeling

from llm_action.src.actions.test import test_action

params_per_kernel = {
    KernelType.MATMUL: {
        "peel_front": False
    }
}

if __name__ == "__main__":
    test_action(Peeling, params_per_kernel)
