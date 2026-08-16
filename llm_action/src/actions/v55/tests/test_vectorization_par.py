from llm_action.src.actions.v55.implementation.vectorization_par import VectorizationPar

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "vector_sizes": [4, 4, 4],
    },
    "add": {
        "vector_sizes": [4, 4, 4, 2],
    },
    "relu": {
        "vector_sizes": [2, 4, 2, 2],
    },
    "pooling_nchw": {
        "vector_sizes": [1, 1, 1, 1, 1, 1],
    },
    "conv_2d_nchw_fchw": {
        "vector_sizes": [1, 1, 1, 1, 1, 1, 1],
    },
}

if __name__ == "__main__":
    test_action(VectorizationPar, params_per_family, benchmark="dataset_ml")
