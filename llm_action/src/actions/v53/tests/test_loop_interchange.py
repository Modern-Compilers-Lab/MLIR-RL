from llm_action.src.actions.v53.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "permutation": [0, 2, 1],  # swap K and N for matmul (3 loops: M, N, K)
    },
    "conv_2d_nchw_fchw": {
        "permutation": [0, 1, 2, 3, 5, 6, 4],  # 7 loops, swap inner dims
    },
    "pooling_nchw": {
        "permutation": [0, 1, 3, 2, 4, 5],  # 6 loops, swap OH and OW
    },
    "add": {
        "permutation": [0, 1, 3, 2],  # 4 loops, swap last two
    },
    "relu": {
        "permutation": [0, 1, 3, 2],  # 4 loops for 4D relu (swap last two)
    },
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_ml")
