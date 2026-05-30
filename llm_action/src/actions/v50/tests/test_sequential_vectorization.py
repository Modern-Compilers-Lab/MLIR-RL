from llm_action.src.actions.v50.implementation.sequential_vectorization import SequentialVectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "vector_sizes": [4, 4, 2, 2, 1, 1],
    }
}

if __name__ == "__main__":
    test_action(SequentialVectorization, params_per_family, benchmark="dataset_pooling")
