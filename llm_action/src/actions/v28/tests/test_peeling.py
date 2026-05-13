from llm_action.src.actions.v28.implementation.peeling import Peeling

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "tile_sizes": [0, 0, 8, 0, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(Peeling, params_per_family, benchmark="paper_conv2d")
