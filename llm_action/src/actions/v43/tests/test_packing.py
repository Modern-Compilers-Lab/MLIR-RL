from llm_action.src.actions.v43.implementation.packing import Packing
from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "packed_sizes": [32, 16, 0, 0, 32, 0, 0]
    }
}

if __name__ == "__main__":
    test_action(Packing, params_per_family, benchmark="dataset_conv2d")
