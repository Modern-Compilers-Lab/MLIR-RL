from llm_action.src.actions.v53.implementation.parallel_vectorization import ParallelVectorization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "vector_sizes": [4, 4, 4],  # 3 loops: M, N, K
    },
    "conv_2d_nchw_fchw": {
        "vector_sizes": [1, 1, 1, 4, 1, 1, 1],  # 7 loops; may fail (windowed access)
    },
    "pooling_nchw": {
        "vector_sizes": [1, 1, 1, 1, 1, 2],  # 6 loops; may fail (windowed access)
    },
    "add": {
        "vector_sizes": [1, 1, 1, 4],  # 4 loops, vectorize innermost
    },
    "relu": {
        "vector_sizes": [1, 4, 1, 1],  # 4 loops for 4D relu; 512%4=0
    },
}

if __name__ == "__main__":
    test_action(ParallelVectorization, params_per_family, benchmark="dataset_ml")
