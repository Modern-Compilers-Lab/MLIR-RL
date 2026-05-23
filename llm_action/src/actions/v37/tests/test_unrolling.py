from llm_action.src.actions.v37.implementation.unrolling import Unrolling

from llm_action.src.actions.test import test_action

params_per_family = {
    "relu": {
        "loop_dim": 0,
        "unroll_factor": 4,
    }
}

if __name__ == "__main__":
    test_action(Unrolling, params_per_family, benchmark="dataset_relu")
