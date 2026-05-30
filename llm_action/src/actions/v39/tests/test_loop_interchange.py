from llm_action.src.actions.v39.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "permutation": [1, 0, 2],
    },
    "conv_2d_nchw_fchw": {
        "permutation": [1, 0, 2, 3, 4, 5, 6],
    },
    "pooling_nchw": {
        "permutation": [1, 0, 2, 3, 4, 5],
    },
    "add": {
        "permutation": [1, 0, 2, 3],
    },
    "relu": {
        "permutation": [1, 0, 2, 3],
    },
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_ml")
