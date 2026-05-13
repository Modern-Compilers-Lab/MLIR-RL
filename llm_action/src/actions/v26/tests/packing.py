from llm_action.src.actions.v26.implementation.packing import Packing

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"packed_sizes": [32, 32, 32]},
    "conv_2d_nchw_fchw": {"packed_sizes": [0, 0, 4, 4, 0, 0, 0]},
    "pooling_nchw": {"packed_sizes": [0, 0, 4, 4, 0, 0]},
    # add/relu omitted: packing is not meaningful for element-wise ops
    # and lower_pack's transpose overhead is excessive on large tensors.
}

if __name__ == "__main__":
    test_action(Packing, params_per_family, benchmark="sample")
