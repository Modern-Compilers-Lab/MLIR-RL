from llm_action.src.actions.v25.implementation.split_reduction import SplitReduction

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "split_factor": 4
    },
}

if __name__ == "__main__":
    test_action(SplitReduction, params_per_family)
