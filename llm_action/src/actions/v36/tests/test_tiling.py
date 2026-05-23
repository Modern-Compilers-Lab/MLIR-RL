from llm_action.src.actions.v36.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "img2col_conv_2d_nchw_fchw": {
        "tile_sizes": [0, 0, 0, 16],
    }
}

if __name__ == "__main__":
    test_action(Tiling, params_per_family, benchmark="dataset_conv2d_img2col")
