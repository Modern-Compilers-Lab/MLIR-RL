from llm_action.src.actions.v49.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "tile_sizes": [0, 32, 0, 0, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_conv2d")
