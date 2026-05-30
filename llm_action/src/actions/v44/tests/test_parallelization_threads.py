from llm_action.src.actions.v44.implementation.parallelization_threads import ParallelizationThreads

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "num_threads": 4,
    },
    "conv_2d_nchw_fchw": {
        "num_threads": 4,
    },
    "pooling_nchw": {
        "num_threads": 4,
    },
    "add": {
        "num_threads": 4,
    },
    "relu": {
        "num_threads": 4,
    },
}

if __name__ == "__main__":
    test_action(ParallelizationThreads, params_per_family, benchmark="dataset_ml")
