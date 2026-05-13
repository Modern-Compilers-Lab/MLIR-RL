from llm_action.src.actions.v24.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 0]
    }
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family)
