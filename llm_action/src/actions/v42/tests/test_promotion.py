from llm_action.src.actions.v42.implementation.promotion import Promotion
from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 32]
    }
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family, benchmark="dataset_matmul")
