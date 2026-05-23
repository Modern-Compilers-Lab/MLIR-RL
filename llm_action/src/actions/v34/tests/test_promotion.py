from llm_action.src.actions.v34.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "tile_sizes": [16, 16, 0, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family, benchmark="dataset_pooling")
