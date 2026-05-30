from llm_action.src.actions.v50.implementation.parallel_vectorization import ParallelVectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "pooling_nchw": {
        "vector_sizes": [4, 4, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(ParallelVectorization, params_per_family, benchmark="dataset_pooling")
