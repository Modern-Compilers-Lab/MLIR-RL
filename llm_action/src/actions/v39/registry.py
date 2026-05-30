from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v39.implementation.tiling import Tiling
from llm_action.src.actions.v39.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v39.implementation.promotion import Promotion
from llm_action.src.actions.v39.implementation.vectorization import Vectorization
from llm_action.src.actions.v39.implementation.parallel_vectorization import ParallelVectorization
from llm_action.src.actions.v39.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v39.implementation.tiling_based_parallelization import TilingBasedParallelization
from llm_action.src.actions.v39.implementation.direct_parallelization import DirectParallelization
from llm_action.src.actions.v39.implementation.im2col_lowering import Im2colLowering

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    ParallelVectorization,
    LoopUnrolling,
    TilingBasedParallelization,
    DirectParallelization,
    Im2colLowering,
]

# Dependency graph for RL action masking.
# Key = action just applied; Value = list of actions that are BLOCKED after it.
# "Blocked" means: applying the blocked action after the key action will fail
# postcondition (or precondition for Im2col cases).
#
# Synthesized from empirical pairwise testing across 5 kernel types:
#   matmul (linalg.matmul), conv (linalg.conv_2d_nchw_fchw),
#   pooling (linalg.pooling_nchw_max), relu (linalg.generic),
#   add (linalg.add)
#
# Universal rules:
#   1. Vectorization/ParallelVectorization are TERMINAL: all 9 actions blocked
#   2. Im2colLowering self-blocks (can't apply twice)
#   3. LoopInterchange blocks Im2colLowering (converts named op to generic)
#   4. All other actions compose freely with each other
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Vectorization",
        "ParallelVectorization",
        "LoopUnrolling",
        "TilingBasedParallelization",
        "DirectParallelization",
        "Im2colLowering",
    ],
    "ParallelVectorization": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Vectorization",
        "ParallelVectorization",
        "LoopUnrolling",
        "TilingBasedParallelization",
        "DirectParallelization",
        "Im2colLowering",
    ],
    "LoopInterchange": [
        "Im2colLowering",
    ],
}
