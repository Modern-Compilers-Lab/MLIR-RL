from llm_action.src.actions.v47.implementation.tiling import Tiling
from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 64, 16],
    }
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_matmul")
