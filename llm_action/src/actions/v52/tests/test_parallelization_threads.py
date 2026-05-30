from llm_action.src.actions.v52.implementation.parallelization_threads import ParallelizationThreads

from llm_action.src.actions.test import test_action

params_per_family = {
    "relu": {
        "num_threads": 4
    }
}

if __name__ == "__main__":
    test_action(ParallelizationThreads, params_per_family, benchmark="dataset_relu")
