from llm_action.src.actions.v39.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [4, 4, 4],
        "operands_to_promote": 0,
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [4, 4, 0, 0, 0, 0, 0],
        "operands_to_promote": 0,
    },
    "pooling_nchw": {
        "tile_sizes": [4, 4, 0, 0, 0, 0],
        "operands_to_promote": 0,
    },
    "add": {
        "tile_sizes": [0, 0, 4, 4],
        "operands_to_promote": 0,
    },
    "relu": {
        "tile_sizes": [4, 4, 0, 0],
        "operands_to_promote": 1,
    },
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family, benchmark="dataset_ml")
