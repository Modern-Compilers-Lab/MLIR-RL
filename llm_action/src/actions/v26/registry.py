from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v26.implementation.tiling import Tiling
from llm_action.src.actions.v26.implementation.promotion import Promotion
from llm_action.src.actions.v26.implementation.packing import Packing
from llm_action.src.actions.v26.implementation.vectorization import Vectorization
from llm_action.src.actions.v26.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v26.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v26.implementation.tiling_based_parallelization import TilingBasedParallelization
from llm_action.src.actions.v26.implementation.thread_count_parallelization import ThreadCountParallelization
from llm_action.src.actions.v26.implementation.split_reduction import SplitReduction
from llm_action.src.actions.v26.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v26.implementation.loop_peeling import LoopPeeling
from llm_action.src.actions.v26.implementation.canonicalization import Canonicalization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    Promotion,
    Packing,
    Vectorization,
    LoopInterchange,
    LoopUnrolling,
    TilingBasedParallelization,
    ThreadCountParallelization,
    SplitReduction,
    Im2colLowering,
    LoopPeeling,
    Canonicalization,
]

# ACTION_DEPENDENCIES encodes which actions become unavailable after
# applying a given action. An edge "X": ["Y"] means that after executing
# action X, action Y will fail (postcondition or precondition failure).
# Only edges that held on EVERY kernel where both endpoints were applicable
# are included. Edges were confirmed via actual MCP tool invocations across
# 20 benchmark kernels (5 families x 4 instances) in v26 exploration.
#
# Terminal actions (block everything except Canonicalization):
#   Vectorization  - lowers linalg to vector dialect
#   Packing        - restructures data layout into complex generics
#   SplitReduction - decomposes reduction into fill + generic pattern
#
# Semi-terminal actions:
#   Im2colLowering - decomposes conv2d; blocks most tested follow-ups
#   Promotion      - converts tensor to memref; blocks parallelization
#   ThreadCountParallelization - blocks Promotion, Vectorization, SplitReduction
#   TilingBasedParallelization - blocks Promotion, SplitReduction
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": [
        "Tiling", "Promotion", "Packing", "Vectorization",
        "LoopInterchange", "LoopUnrolling",
        "TilingBasedParallelization", "ThreadCountParallelization",
        "SplitReduction", "LoopPeeling",
    ],
    "Packing": [
        "Tiling", "Promotion", "Packing", "Vectorization",
        "LoopInterchange", "LoopUnrolling",
        "TilingBasedParallelization", "ThreadCountParallelization",
        "SplitReduction", "LoopPeeling",
    ],
    "SplitReduction": [
        "Tiling", "Promotion", "Packing", "Vectorization",
        "LoopInterchange", "LoopUnrolling",
        "TilingBasedParallelization", "ThreadCountParallelization",
        "SplitReduction", "LoopPeeling",
    ],
    "Im2colLowering": [
        "Tiling", "Vectorization", "LoopUnrolling",
        "TilingBasedParallelization", "ThreadCountParallelization",
        "SplitReduction", "Im2colLowering",
    ],
    "Promotion": [
        "TilingBasedParallelization", "ThreadCountParallelization",
    ],
    "ThreadCountParallelization": [
        "Promotion", "Vectorization", "SplitReduction",
    ],
    "TilingBasedParallelization": [
        "Promotion", "SplitReduction",
    ],
}
