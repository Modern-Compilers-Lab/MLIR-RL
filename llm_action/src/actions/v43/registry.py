from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v43.implementation.tiling import Tiling
from llm_action.src.actions.v43.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v43.implementation.promotion import Promotion
from llm_action.src.actions.v43.implementation.packing import Packing
from llm_action.src.actions.v43.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v43.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v43.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v43.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v43.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v43.implementation.parallelization_threads import ParallelizationThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Packing,
    VectorizationSequential,
    VectorizationParallel,
    LoopUnrolling,
    Im2colLowering,
    ParallelizationTiling,
    ParallelizationThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "VectorizationSequential": [
        "Tiling", "LoopInterchange", "Promotion", "Packing",
        "VectorizationSequential", "VectorizationParallel",
        "LoopUnrolling", "Im2colLowering",
        "ParallelizationTiling", "ParallelizationThreads",
    ],
    "VectorizationParallel": [
        "Tiling", "LoopInterchange", "Promotion", "Packing",
        "VectorizationSequential", "VectorizationParallel",
        "LoopUnrolling", "Im2colLowering",
        "ParallelizationTiling", "ParallelizationThreads",
    ],
    "LoopInterchange": [
        "Packing", "VectorizationSequential",
        "VectorizationParallel", "Im2colLowering",
    ],
    "Promotion": ["VectorizationParallel", "VectorizationSequential"],
    "Packing": ["VectorizationParallel"],
}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "conv_2d_nchw_fchw": [
        ["ParallelizationTiling", "Tiling", "Promotion"],
        ["ParallelizationTiling", "Tiling", "VectorizationParallel"],
        ["ParallelizationTiling", "Tiling", "LoopUnrolling"],
        ["Im2colLowering", "ParallelizationTiling", "VectorizationParallel"],
        ["ParallelizationTiling", "VectorizationParallel"],
        ["ParallelizationTiling", "Tiling", "Tiling", "VectorizationParallel"],
    ],
}
