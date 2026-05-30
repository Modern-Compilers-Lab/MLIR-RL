from llm_action.src.actions.v49.implementation.vectorization_seq import VectorizationSeq

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "vector_sizes": [1, 4, 1, 4],
    }
}

if __name__ == "__main__":
    test_action(VectorizationSeq, params_per_family, benchmark="dataset_conv2d")
