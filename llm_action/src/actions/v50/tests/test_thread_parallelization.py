from llm_action.src.actions.v50.implementation.thread_parallelization import ThreadParallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "num_threads": 4,
    }
}

if __name__ == "__main__":
    test_action(ThreadParallelization, params_per_family, benchmark="dataset_pooling")
