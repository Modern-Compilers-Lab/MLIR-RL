from llm_action.src.actions.v55.implementation.parallelization_tiling import ParallelizationTiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32],
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [32],
    },
    "pooling_nchw": {
        "tile_sizes": [32],
    },
    "add": {
        "tile_sizes": [16, 16],
    },
    "relu": {
        "tile_sizes": [32, 32],
    },
}

if __name__ == "__main__":
    test_action(ParallelizationTiling, params_per_family, benchmark="dataset_ml")
