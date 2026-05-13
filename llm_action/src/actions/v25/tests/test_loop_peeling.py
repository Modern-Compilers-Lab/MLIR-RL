from llm_action.src.actions.v25.implementation.loop_peeling import LoopPeeling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_size": 10
    },
}

if __name__ == "__main__":
    test_action(LoopPeeling, params_per_family)
