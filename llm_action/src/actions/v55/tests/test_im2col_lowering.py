from llm_action.src.actions.v55.implementation.im2col_lowering import Im2colLowering

from llm_action.src.actions.test import test_action

params_per_family = {
    "conv_2d_nchw_fchw": {},
    "matmul": {},
    "pooling_nchw": {},
    "add": {},
    "relu": {},
}

if __name__ == "__main__":
    test_action(Im2colLowering, params_per_family, benchmark="dataset_ml")
