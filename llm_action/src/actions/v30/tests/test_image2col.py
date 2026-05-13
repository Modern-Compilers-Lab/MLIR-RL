from llm_action.src.actions.v30.implementation.image2col import Image2Col

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {},
}

if __name__ == "__main__":
    test_action(Image2Col, params_per_family, benchmark="dataset_conv2d")
