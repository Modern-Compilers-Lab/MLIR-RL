from llm_action.src.actions.v27.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.test import test_action

params_per_family = {
    "matmul": {
        "tile_sizes": [32, 32, 0],
    }
}

if __name__ == "__main__":
    test_action(ParallelizationTiling, params_per_family, benchmark="paper_matmul")
