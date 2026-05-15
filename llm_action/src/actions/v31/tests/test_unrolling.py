from llm_action.src.actions.v31.implementation.unrolling import Unrolling

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "tile_sizes": [0, 0, 8, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_family, benchmark="dataset_pooling")
