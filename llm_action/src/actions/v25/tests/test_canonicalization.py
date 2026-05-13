from llm_action.src.actions.v25.implementation.canonicalization import Canonicalization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {},
}

if __name__ == "__main__":
    test_action(Canonicalization, params_per_family)
