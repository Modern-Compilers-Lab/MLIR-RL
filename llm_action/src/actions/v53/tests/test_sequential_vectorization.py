from llm_action.src.actions.v53.implementation.sequential_vectorization import SequentialVectorization

from llm_action.src.actions.test import test_action

# Vectorization is a lowering transform. conv2d and pooling have complex
# access patterns that resist direct vectorization — im2col or generalize first.
# For these families, use small vector sizes that divide all dims.
params_per_family = {
    "matmul": {
        "vector_sizes": [4, 4, 4],  # 3 loops: M, N, K
    },
    "conv_2d_nchw_fchw": {
        "vector_sizes": [1, 1, 1, 4, 1, 1, 1],  # 7 loops; conv may fail due to windowed access
    },
    "pooling_nchw": {
        "vector_sizes": [1, 1, 1, 1, 1, 2],  # 6 loops; pooling may fail due to windowed access
    },
    "add": {
        "vector_sizes": [1, 1, 1, 4],  # 4 loops, vectorize innermost
    },
    "relu": {
        "vector_sizes": [1, 4, 1, 1],  # 4 loops for 4D relu; 512%4=0
    },
}

if __name__ == "__main__":
    test_action(SequentialVectorization, params_per_family, benchmark="dataset_ml")
