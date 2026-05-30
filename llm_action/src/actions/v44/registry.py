from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v44.implementation.tiling import Tiling
from llm_action.src.actions.v44.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v44.implementation.packing import Packing
from llm_action.src.actions.v44.implementation.promotion import Promotion
from llm_action.src.actions.v44.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v44.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v44.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v44.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v44.implementation.parallelization_threads import ParallelizationThreads
from llm_action.src.actions.v44.implementation.im2col_lowering import Im2colLowering

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Packing,
    Promotion,
    VectorizationSequential,
    VectorizationParallel,
    LoopUnrolling,
    ParallelizationTiling,
    ParallelizationThreads,
    Im2colLowering,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # VecSeq fails Post on conv2d and pooling structured ops
    "VectorizationSequential": ["conv_2d_nchw_fchw", "pooling_nchw_max"],
    # Im2col only works on conv2d
    "Im2colLowering": ["matmul", "pooling_nchw_max", "add", "relu", "generic"],
}

# Per-family allowlist of schedule paths (action-sequence skeletons).
# RL policy tunes parameters; these define the valid action orderings.
# Terminal actions (VecSeq, VecPar, ParallelTiling, ParallelThreads) must be last.
SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "matmul": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "VectorizationSequential"],
        ["Tiling", "Promotion", "VectorizationSequential"],
        ["Tiling", "Promotion", "ParallelizationTiling"],
        ["VectorizationSequential"],
        ["ParallelizationThreads"],
        ["Tiling", "ParallelizationThreads"],
        ["Promotion", "ParallelizationTiling"],
        ["LoopInterchange", "ParallelizationTiling"],
    ],
    "conv2d": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Im2colLowering", "ParallelizationTiling"],
        ["Im2colLowering", "Tiling", "ParallelizationTiling"],
        ["Tiling", "ParallelizationThreads"],
        ["ParallelizationThreads"],
    ],
    "pooling": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "ParallelizationThreads"],
        ["ParallelizationThreads"],
    ],
    "add": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "VectorizationSequential"],
        ["VectorizationSequential"],
        ["ParallelizationThreads"],
    ],
    "relu": [
        ["ParallelizationTiling"],
        ["Tiling", "ParallelizationTiling"],
        ["Tiling", "VectorizationSequential"],
        ["VectorizationSequential"],
        ["ParallelizationThreads"],
    ],
}
