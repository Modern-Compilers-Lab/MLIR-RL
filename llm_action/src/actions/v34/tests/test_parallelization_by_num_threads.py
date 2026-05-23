from llm_action.src.actions.v34.implementation.parallelization_by_num_threads import ParallelizationByNumThreads

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "num_threads": 4,
    }
}

if __name__ == "__main__":
    test_action(ParallelizationByNumThreads, params_per_family, benchmark="dataset_pooling")
