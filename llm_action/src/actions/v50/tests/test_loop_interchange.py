from llm_action.src.actions.v50.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "permutation": [0, 1, 3, 2, 4, 5],
    }
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_pooling")
