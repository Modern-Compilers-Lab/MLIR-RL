from llm_action.src.actions.v38.implementation.tiling_based_parallelization import TilingBasedParallelization

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [16, 16, 0]
    }
}

if __name__ == "__main__":
    test_action(TilingBasedParallelization, params_per_family, benchmark="dataset_matmul")
