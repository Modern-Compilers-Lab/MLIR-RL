from llm_action.src.models import KernelType
from llm_action.src.actions.v6.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

# matmul has 3 loops (M=128, K=256, N=128)
# conv2d has 7 loops (N=128, F=256, OH=7, OW=7, C=32, KH=1, KW=1)
# generic has 5 loops (a=8, b=8, c=16, d=8, e=32)
params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 32, 0],
    },
    KernelType.CONV2D: {
        "tile_sizes": [32, 32, 0, 0, 0, 0, 0],
    },
    KernelType.GENERIC: {
        "tile_sizes": [4, 4, 0, 0, 0],
    },
}

if __name__ == "__main__":
    test_action(Tiling, params_per_kernel)
