from llm_action.src.actions.v34.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "tile_sizes": [4, 4, 4, 4, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_family, benchmark="dataset_pooling")
