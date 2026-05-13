from llm_action.src.actions.v26.implementation.loop_peeling import LoopPeeling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"tile_size": 48},
    "conv_2d_nchw_fchw": {"tile_size": 48},
    "pooling_nchw": {"tile_size": 48},
    "add": {"tile_size": 48},
    "relu": {"tile_size": 48},
}

if __name__ == "__main__":
    test_action(LoopPeeling, params_per_family, benchmark="sample")
