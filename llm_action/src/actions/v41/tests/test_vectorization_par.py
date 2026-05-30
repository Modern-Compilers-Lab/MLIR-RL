from llm_action.src.actions.v41.implementation.vectorization_par import VectorizationPar
from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "vector_sizes": [4, 4, 4],
    }
}

if __name__ == "__main__":
    test_action(VectorizationPar, params_per_family, benchmark="dataset_matmul")
