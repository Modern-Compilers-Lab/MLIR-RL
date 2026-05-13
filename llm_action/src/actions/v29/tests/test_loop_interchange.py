from llm_action.src.actions.v29.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "permutation": [1, 0, 2]
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_matmul")
