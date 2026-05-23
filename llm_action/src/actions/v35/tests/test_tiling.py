from llm_action.src.actions.v35.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "add": {
        "tile_sizes": [8, 8, 0, 0]
    }
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_add")
