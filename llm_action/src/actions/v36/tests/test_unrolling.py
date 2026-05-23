from llm_action.src.actions.v36.implementation.unrolling import Unrolling

from llm_action.src.actions.test import test_action

params_per_family = {
    "img2col_conv_2d_nchw_fchw": {
        "loop_dim": 3,
        "unroll_factor": 4,
    }
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_family, benchmark="dataset_conv2d_img2col")
