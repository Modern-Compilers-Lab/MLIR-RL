from llm_action.src.actions.v41.implementation.unrolling import Unrolling
from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [4, 0, 0],
        "unroll_factor": 4,
    }
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_family, benchmark="dataset_matmul")
