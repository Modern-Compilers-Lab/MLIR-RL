from llm_action.src.actions.v46.implementation.vectorization_sequential import VectorizationSequential

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "vector_sizes": [4, 4, 4]
    }
}

if __name__ == "__main__":
    test_action(VectorizationSequential, params_per_family, benchmark="dataset_matmul")
