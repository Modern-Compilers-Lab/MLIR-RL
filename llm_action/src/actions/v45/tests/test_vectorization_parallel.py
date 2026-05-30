from llm_action.src.actions.v45.implementation.vectorization_parallel import VectorizationParallel

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [8, 8, 4],
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [4, 4, 2, 2, 4, 0, 0],
    },
    "pooling_nchw": {
        "tile_sizes": [16, 16, 1, 1, 1, 1],
    },
    "add": {
        "tile_sizes": [1, 2, 4, 4],
    },
    "relu": {
        "tile_sizes": [4, 4, 1, 1],
    },
}

if __name__ == "__main__":
    test_action(VectorizationParallel, params_per_family, benchmark="dataset_ml")
