from llm_action.src.actions.v43.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "vector_sizes": [1, 1, 1, 4, 1, 1, 1]
    }
}

if __name__ == "__main__":
    test_action(VectorizationSequential, params_per_family, benchmark="dataset_conv2d")
