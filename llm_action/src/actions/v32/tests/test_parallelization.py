from llm_action.src.actions.v32.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"num_threads": [28, 1]}
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_family, benchmark="dataset_matmul")
