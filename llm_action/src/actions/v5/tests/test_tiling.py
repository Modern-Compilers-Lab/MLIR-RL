from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.tiling import Tiling

from llm_action.src.actions.test import test_action

# MATMUL: 3 iteration dims (M=128, N=128, K=256)
# CONV2D: 7 iteration dims (N=128, F=256, OH=7, OW=7, C=32, KH=1, KW=1)
# GENERIC: 5 iteration dims (all parallel, dims 8,8,16,8,32)
params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 32, 0],
    },
}

if __name__ == "__main__":
    test_action(Tiling, params_per_kernel)
