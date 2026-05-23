from llm_action.src.actions.v37.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "relu": {
        "tile_sizes": [1, 1, 5, 5],
    }
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_family, benchmark="dataset_relu")
