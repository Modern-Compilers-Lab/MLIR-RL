from llm_action.src.actions.v32.implementation.parallel_tiling import ParallelTiling

from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {"tile_sizes": [64, 64, 0]}
}

if __name__ == "__main__":
    test_action(ParallelTiling, params_per_family, benchmark="dataset_matmul")
