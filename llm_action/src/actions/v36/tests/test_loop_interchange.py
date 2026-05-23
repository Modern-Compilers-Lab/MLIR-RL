from llm_action.src.actions.v36.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "img2col_conv_2d_nchw_fchw": {
        "permutation": [3, 1, 2, 0],
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_conv2d_img2col")
