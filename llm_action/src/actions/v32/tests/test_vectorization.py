from llm_action.src.actions.v32.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"tile_sizes": [8, 4, 8]}
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_family, benchmark="dataset_matmul")
