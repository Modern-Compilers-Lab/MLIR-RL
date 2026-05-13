from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v29.implementation.tiling import Tiling
from llm_action.src.actions.v29.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v29.implementation.promotion import Promotion
from llm_action.src.actions.v29.implementation.vectorization import Vectorization
from llm_action.src.actions.v29.implementation.unrolling import Unrolling
from llm_action.src.actions.v29.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v29.implementation.parallelization_direct import ParallelizationDirect

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Unrolling,
    ParallelizationTiling,
    ParallelizationDirect,
]

# Empirically determined from exhaustive pairwise composition testing on 11 matmul kernels.
# "terminal" = after this action, ALL 7 actions fail postcondition (pre=T, post=F).
# Vectorization replaces linalg ops with vector dialect ops; Unrolling restructures
# the loop nest such that subsequent actions find the tag but cannot transform.
# All other 5 actions (Tiling, LoopInterchange, Promotion, ParTiling, ParDirect)
# compose freely with all 7 actions in any order.
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Vectorization",
        "Unrolling",
        "ParallelizationTiling",
        "ParallelizationDirect",
    ],
    "Unrolling": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Vectorization",
        "Unrolling",
        "ParallelizationTiling",
        "ParallelizationDirect",
    ],
}
