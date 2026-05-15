from llm_action.src.actions.v31.implementation.parallelization_tiling import ParallelizationTiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "tile_sizes": [16, 0, 0, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(ParallelizationTiling, params_per_family, benchmark="dataset_pooling")
