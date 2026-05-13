from llm_action.src.actions.v25.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "operands_to_promote": [0, 1, 2]
    },
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family)
