from llm_action.src.actions.v51.implementation.parallelization_tile import ParallelizationTile

from llm_action.src.actions.test import test_action

params_per_family = {
    "add": {
        "tile_sizes": [16, 16, 0, 0]
    }
}

if __name__ == "__main__":
    test_action(ParallelizationTile, params_per_family, benchmark="dataset_add")
