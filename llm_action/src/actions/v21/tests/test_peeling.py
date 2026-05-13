from llm_action.src.actions.v21.implementation.peeling import Peeling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_size": 48
    }
}

if __name__ == "__main__":
    test_action(Peeling, params_per_family)
