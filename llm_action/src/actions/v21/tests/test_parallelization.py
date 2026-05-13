from llm_action.src.actions.v21.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "num_threads": [4, 4]
    }
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_family)
