from llm_action.src.actions.v53.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 32],
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [0, 0, 0, 0, 32, 0, 0],  # tile C dim
    },
    "pooling_nchw": {
        "tile_sizes": [0, 0, 0, 8, 0, 0],  # tile OW dim
    },
    "add": {
        "tile_sizes": [0, 0, 0, 16],  # tile last dim
    },
    "relu": {
        "tile_sizes": [0, 4, 0, 0],  # tile C dim for 4D relu
    },
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family, benchmark="dataset_ml")
