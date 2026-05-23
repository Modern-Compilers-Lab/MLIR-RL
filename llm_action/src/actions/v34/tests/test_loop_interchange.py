from llm_action.src.actions.v34.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "permutation": [2, 3, 0, 1, 4, 5],
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_pooling")
