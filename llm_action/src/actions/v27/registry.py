from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v27.implementation.tiling import Tiling
from llm_action.src.actions.v27.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v27.implementation.promotion import Promotion
from llm_action.src.actions.v27.implementation.vectorization import Vectorization
from llm_action.src.actions.v27.implementation.unrolling import Unrolling
from llm_action.src.actions.v27.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v27.implementation.parallelization_direct import ParallelizationDirect

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Unrolling,
    ParallelizationTiling,
    ParallelizationDirect,
]

# Cross-kernel composability analysis (v27, matmul family, 4 kernels).
# Maps each action to the list of actions FORBIDDEN after it
# (postcondition = False on the subsequent action).
#
# Empirical findings (all 49 pairs tested on matmul_256_256_128):
#   - Vectorization is TERMINAL: replaces linalg op with vector ops,
#     so no subsequent action can match the tagged operation (Post=F).
#   - Unrolling is TERMINAL: creates multiple tagged linalg op copies
#     in the loop body; subsequent actions all return Post=F.
#   - Tiling, LoopInterchange, Promotion, ParallelizationTiling,
#     and ParallelizationDirect compose freely with ALL actions (Pre=T, Post=T).
#   - Execution caveat: Vectorization after Tiling may fail at runtime
#     when tile sizes do not evenly divide all dimensions (dynamic masking).
#     This is an execution-time issue, not a composability issue.
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Tiling": [],
    "LoopInterchange": [],
    "Promotion": [],
    "Vectorization": [
        "Tiling", "LoopInterchange", "Promotion",
        "Vectorization", "Unrolling",
        "ParallelizationTiling", "ParallelizationDirect",
    ],
    "Unrolling": [
        "Tiling", "LoopInterchange", "Promotion",
        "Vectorization", "Unrolling",
        "ParallelizationTiling", "ParallelizationDirect",
    ],
    "ParallelizationTiling": [],
    "ParallelizationDirect": [],
}
