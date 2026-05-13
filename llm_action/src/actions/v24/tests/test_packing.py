from llm_action.src.actions.v24.implementation.packing import Packing

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "packed_sizes": [32, 32, 0]
    }
}

if __name__ == "__main__":
    test_action(Packing, params_per_family)
