from llm_action.src.actions.v33.implementation.parallel_tiling import ParallelTiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "tile_sizes": [4, 4, 0, 0, 0, 0, 0]
    }
}

if __name__ == "__main__":
    test_action(ParallelTiling, params_per_family, benchmark="dataset_conv2d")
