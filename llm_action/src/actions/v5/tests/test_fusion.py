from llm_action.src.models import KernelType
from llm_action.src.actions.v5.implementation.fusion import Fusion

from llm_action.src.actions.test import test_action

# Fuse tiles the operation and fuses producers into the loop
# Similar parameters to tiling
params_per_kernel = {
    KernelType.MATMUL: {
        "tile_sizes": [32, 32, 0],
    },
}

if __name__ == "__main__":
    test_action(Fusion, params_per_kernel)
