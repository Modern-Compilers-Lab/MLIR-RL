from llm_action.src.actions.v32.implementation.loop_interchange import LoopInterchange

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"permutation": [2, 0, 1]}
}

if __name__ == "__main__":
    test_action(LoopInterchange, params_per_family, benchmark="dataset_matmul")
