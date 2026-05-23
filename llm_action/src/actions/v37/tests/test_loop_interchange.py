from llm_action.src.actions.v37.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "relu": {
        "permutation": [2, 3, 0, 1],
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_relu")
