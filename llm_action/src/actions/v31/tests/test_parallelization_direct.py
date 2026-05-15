from llm_action.src.actions.v31.implementation.parallelization_direct import ParallelizationDirect

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "num_threads": [4, 0, 0, 0, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(ParallelizationDirect, params_per_family, benchmark="dataset_pooling")
