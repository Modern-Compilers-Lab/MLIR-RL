from llm_action.src.actions.v33.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "num_threads": 4
    }
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_family, benchmark="dataset_conv2d")
