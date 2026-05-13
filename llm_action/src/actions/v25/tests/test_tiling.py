from llm_action.src.actions.v25.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 16]
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [4, 8, 0, 0, 0, 0, 0]
    },
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family)
