from llm_action.src.actions.v42.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "permutation": [0, 2, 1]
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_matmul")
