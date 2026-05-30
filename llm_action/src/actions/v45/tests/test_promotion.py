from llm_action.src.actions.v45.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 32],
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [4, 4, 2, 2, 4, 0, 0],
    },
    "pooling_nchw": {
        "tile_sizes": [16, 16, 0, 0, 0, 0],
    },
    "add": {
        "tile_sizes": [0, 0, 32, 32],
    },
    "relu": {
        "tile_sizes": [0, 0, 0, 0],
    },
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family, benchmark="dataset_ml")
