from llm_action.src.actions.v30.implementation.unrolling import Unrolling

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "tile_sizes": [0, 0, 0, 0, 4, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_family, benchmark="dataset_conv2d")
