from llm_action.src.actions.v37.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "relu": {
        "tile_sizes": [32, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_family, benchmark="dataset_relu")
