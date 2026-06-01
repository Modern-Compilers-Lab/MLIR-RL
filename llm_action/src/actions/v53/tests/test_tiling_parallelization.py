from llm_action.src.actions.v53.implementation.tiling_parallelization import TilingParallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 0],  # tile M and N parallel dims, not K (reduction)
    },
    "conv_2d_nchw_fchw": {
        "tile_sizes": [32, 32, 0, 0, 0, 0, 0],  # tile N and F (parallel)
    },
    "pooling_nchw": {
        "tile_sizes": [32, 32, 0, 0, 0, 0],  # tile N and C (parallel)
    },
    "add": {
        "tile_sizes": [0, 0, 0, 16],  # tile last dim
    },
    "relu": {
        "tile_sizes": [64, 0, 0, 0],  # tile first dim
    },
}

if __name__ == "__main__":
    test_action(TilingParallelization, params_per_family, benchmark="dataset_ml")
