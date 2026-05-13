from llm_action.src.actions.v22.implementation.loop_peeling import LoopPeeling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {}
}

if __name__ == "__main__":
    test_action(LoopPeeling, params_per_family)
