from llm_action.src.actions.v52.implementation.vectorization_parallel import VectorizationParallel

from llm_action.src.actions.test import test_action

params_per_family = {
    "relu": {
        "vector_sizes": [1, 4]
    }
}

if __name__ == "__main__":
    test_action(VectorizationParallel, params_per_family, benchmark="dataset_relu")
