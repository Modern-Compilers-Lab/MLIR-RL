from llm_action.src.actions.v43.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "tile_sizes": [32, 0, 0, 0, 0, 0, 0]
    }
}

if __name__ == "__main__":
    test_action(ParallelizationTiling, params_per_family, benchmark="dataset_conv2d")
