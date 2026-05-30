from llm_action.src.actions.v43.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "unroll_factor": 4
    }
}

if __name__ == "__main__":
    test_action(LoopUnrolling, params_per_family, benchmark="dataset_conv2d")
