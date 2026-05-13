from llm_action.src.actions.v28.implementation.parallelization_direct import ParallelizationDirect

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "num_threads": [4, 0, 0, 0, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(ParallelizationDirect, params_per_family, benchmark="paper_conv2d")
