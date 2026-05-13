from llm_action.src.actions.v26.implementation.split_reduction import SplitReduction

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"split_factor": 4},
    "conv_2d_nchw_fchw": {"split_factor": 4},
    "pooling_nchw": {"split_factor": 4},
}

if __name__ == "__main__":
    test_action(SplitReduction, params_per_family, benchmark="sample")
