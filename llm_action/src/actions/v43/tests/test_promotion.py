from llm_action.src.actions.v43.implementation.promotion import Promotion
from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {
        "operands_to_promote": [0, 1, 2]
    }
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family, benchmark="dataset_conv2d")
