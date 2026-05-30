from llm_action.src.actions.v46.implementation.vectorization_parallel import VectorizationParallel

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [64, 64, 4]
    }
}

if __name__ == "__main__":
    test_action(VectorizationParallel, params_per_family, benchmark="dataset_matmul")
