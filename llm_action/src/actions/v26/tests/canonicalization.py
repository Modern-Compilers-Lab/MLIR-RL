from llm_action.src.actions.v26.implementation.canonicalization import Canonicalization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {},
    "conv_2d_nchw_fchw": {},
    "pooling_nchw": {},
    "add": {},
    "relu": {},
}

if __name__ == "__main__":
    test_action(Canonicalization, params_per_family, benchmark="sample")
