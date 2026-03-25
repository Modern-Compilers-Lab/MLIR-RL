from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.parallelization import Parallelization

from llm_action.src.actions.test import test_action

# tile_sizes for forall parallelization (only parallel dims)
# MATMUL: M=128, N=128 are parallel -> tile_sizes [32, 32]
# CONV2D: N=128, F=256, OH=7, OW=7 are parallel -> tile first two
# GENERIC: all 5 dims are parallel -> tile first two
params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 32],
    },
}

if __name__ == "__main__":
    test_action(Parallelization, params_per_kernel)
