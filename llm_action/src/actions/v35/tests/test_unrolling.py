from llm_action.src.actions.v35.implementation.unrolling import Unrolling

from llm_action.src.actions.test import test_action

params_per_family = {
    "add": {
        "loop_dim": 0,
        "unroll_factor": 4
    }
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_family, benchmark="dataset_add")
