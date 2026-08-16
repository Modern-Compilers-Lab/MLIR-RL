from llm_action.src.actions.v55.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 0],
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [0, 0, 0, 0, 8, 0, 0],
    },
    "pooling_nchw": {
        "tile_sizes": [0, 0, 4, 4, 0, 0],
    },
    "add": {
        "tile_sizes": [16, 16, 0, 0],
    },
    "relu": {
        "tile_sizes": [16, 16],
    },
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_ml")
