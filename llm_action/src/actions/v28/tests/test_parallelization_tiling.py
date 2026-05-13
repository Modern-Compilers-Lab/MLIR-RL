from llm_action.src.actions.v28.implementation.parallelization_tiling import ParallelizationTiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "tile_sizes": [4, 32, 0, 0, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(ParallelizationTiling, params_per_family, benchmark="paper_conv2d")
