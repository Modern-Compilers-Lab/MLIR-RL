from llm_action.src.actions.v44.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 0],
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [0, 16, 0, 0, 0, 0, 0],
    },
    "pooling_nchw": {
        "tile_sizes": [0, 0, 4, 4, 0, 0],
    },
    "add": {
        "tile_sizes": [0, 0, 4, 4],
    },
    "relu": {
        "tile_sizes": [32, 0, 0, 0],
    },
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_ml")
