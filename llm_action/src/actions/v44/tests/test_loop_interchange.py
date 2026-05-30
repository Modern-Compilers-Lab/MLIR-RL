from llm_action.src.actions.v44.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "permutation": [0, 2, 1],
    },
    "conv_2d_nchw_fchw": {
        "permutation": [1, 0, 2, 3, 4, 5, 6],
    },
    "pooling_nchw": {
        "permutation": [1, 0, 2, 3, 4, 5],
    },
    "add": {
        "permutation": [0, 1, 3, 2],
    },
    "relu": {
        "permutation": [0, 1, 3, 2],
    },
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_ml")
