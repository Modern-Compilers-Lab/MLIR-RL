from llm_action.src.actions.v0.implementation.name import Name

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        ...
    }
}

if __name__ == "__main__":
    test_action(Name, params_per_family)
