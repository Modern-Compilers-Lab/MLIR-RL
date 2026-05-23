from llm_action.src.actions.v32.implementation.packing import Packing

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"packed_sizes": [0, 0, 32]}
}

if __name__ == "__main__":
    test_action(Packing, params_per_family, benchmark="dataset_matmul")
