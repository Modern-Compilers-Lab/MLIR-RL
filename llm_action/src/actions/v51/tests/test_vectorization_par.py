from llm_action.src.actions.v51.implementation.vectorization_par import VectorizationPar

from llm_action.src.actions.test import test_action

params_per_family = {
    "add": {
        "vector_sizes": [4, 4, 1, 2]
    }
}

if __name__ == "__main__":
    test_action(VectorizationPar, params_per_family, benchmark="dataset_add")
