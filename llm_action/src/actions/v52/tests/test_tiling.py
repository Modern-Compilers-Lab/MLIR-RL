from llm_action.src.actions.v52.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "relu": {
        "tile_sizes": [32, 64]
    }
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_relu")
