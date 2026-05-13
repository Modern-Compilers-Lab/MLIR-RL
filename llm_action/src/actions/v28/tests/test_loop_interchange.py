from llm_action.src.actions.v28.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "permutation": [0, 1, 4, 2, 3, 5, 6],
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="paper_conv2d")
