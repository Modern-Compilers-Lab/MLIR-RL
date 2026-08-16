from llm_action.src.actions.v55.implementation.packing import Packing

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "packed_sizes": [4, 4, 4],
    },
    "conv_2d_nchw_fchw": {
        "packed_sizes": [0, 0, 0, 0, 4, 0, 0],
    },
    "pooling_nchw": {
        "packed_sizes": [0, 4, 0, 0, 0, 0],
    },
    "add": {
        "packed_sizes": [4, 4, 0, 0],
    },
    "relu": {
        "packed_sizes": [4, 4, 0, 0],
    },
}

if __name__ == "__main__":
    test_action(Packing, params_per_family, benchmark="dataset_ml")
