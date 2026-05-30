from llm_action.src.actions.v39.implementation.parallel_vectorization import ParallelVectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "vector_sizes": [4, 4, 4],
    },
    "conv_2d_nchw_fchw": {
        "vector_sizes": [1, 4, 4, 4, 4, 1, 1],
    },
    "pooling_nchw": {
        "vector_sizes": [1, 4, 1, 1, 1, 1],
    },
    "add": {
        "vector_sizes": [1, 2, 4, 4],
    },
    "relu": {
        "vector_sizes": [4, 4, 1, 1],
    },
}

if __name__ == "__main__":
    test_action(ParallelVectorization, params_per_family, benchmark="dataset_ml")
