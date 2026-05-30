from llm_action.src.actions.v41.implementation.parallelization_tile import ParallelizationTile
from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 0, 0],
    }
}

if __name__ == "__main__":
    test_action(ParallelizationTile, params_per_family, benchmark="dataset_matmul")
