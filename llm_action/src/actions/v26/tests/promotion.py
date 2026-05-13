from llm_action.src.actions.v26.implementation.promotion import Promotion

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"operands_to_promote": [0, 1, 2]},
    "conv_2d_nchw_fchw": {"operands_to_promote": [0, 1]},
    "pooling_nchw": {"operands_to_promote": [0, 1, 2]},
    "add": {"operands_to_promote": [0, 1]},
    "relu": {"operands_to_promote": [0, 1]},
}

if __name__ == "__main__":
    test_action(Promotion, params_per_family, benchmark="sample")
