from llm_action.src.actions.v25.implementation.thread_count_parallelization import ThreadCountParallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "num_threads": 4
    },
}

if __name__ == "__main__":
    test_action(ThreadCountParallelization, params_per_family)
