from llm_action.src.actions.v38.implementation.direct_parallelization import DirectParallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "num_threads": 4
    }
}

if __name__ == "__main__":
    test_action(DirectParallelization, params_per_family, benchmark="dataset_matmul")
