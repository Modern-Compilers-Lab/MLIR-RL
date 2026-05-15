from llm_action.src.actions.v31.implementation.vectorization import Vectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "vector_sizes": [1, 1, 1, 4, 1, 1],
    }
}

if __name__ == "__main__":
    test_action(Vectorization, params_per_family, benchmark="dataset_pooling")
