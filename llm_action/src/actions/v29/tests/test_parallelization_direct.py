from llm_action.src.actions.v29.implementation.parallelization_direct import ParallelizationDirect

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "num_threads": [4, 4, 0]
    }
}

if __name__ == "__main__":
    test_action(ParallelizationDirect, params_per_family, benchmark="dataset_matmul")
