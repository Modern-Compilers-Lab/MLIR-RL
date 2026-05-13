from llm_action.src.actions.v30.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "permutation": [1, 0, 2, 3, 4, 5, 6],
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_conv2d")
