from llm_action.src.actions.v53.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 32],
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [0, 0, 0, 0, 0, 0, 0],
    },
    "pooling_nchw": {
        "tile_sizes": [0, 0, 0, 0, 0, 0],
    },
    "add": {
        "tile_sizes": [0, 0, 0, 16],
    },
    "relu": {
        "tile_sizes": [0, 4],
    },
}

# Use divisible tile sizes for conv2d - 7 loops: N, F, OH, OW, C, KH, KW
# For the example instance 128x128x14x14 -> 192x128x1x1 -> 128x192x7x7
# Loops: N=128, F=192, OH=7, OW=7, C=128, KH=1, KW=1
# Use 0 for dims we don't tile, 4 for those we do
params_per_family["conv_2d_nchw_fchw"] = {"tile_sizes": [0, 0, 0, 0, 32, 0, 0]}

# For pooling: 6 loops: N, C, OH, OW, KH, KW
params_per_family["pooling_nchw"] = {"tile_sizes": [0, 0, 0, 8, 0, 0]}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_ml")
