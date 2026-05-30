from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v49.implementation.tiling import Tiling
from llm_action.src.actions.v49.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v49.implementation.promotion import Promotion
from llm_action.src.actions.v49.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v49.implementation.vectorization_seq import VectorizationSeq
from llm_action.src.actions.v49.implementation.vectorization_par import VectorizationPar
from llm_action.src.actions.v49.implementation.parallelization_tile import ParallelizationTile
from llm_action.src.actions.v49.implementation.parallelization_threads import ParallelizationThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Im2colLowering,
    VectorizationSeq,
    VectorizationPar,
    ParallelizationTile,
    ParallelizationThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "LoopInterchange": ["Im2colLowering", "VectorizationSeq", "VectorizationPar"],
    "Promotion": ["Im2colLowering", "VectorizationSeq", "VectorizationPar"],
    "Im2colLowering": ["Im2colLowering"],
    "VectorizationSeq": ["Im2colLowering"],
    "VectorizationPar": ["Im2colLowering"],
}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "conv_2d_nchw_fchw": [
        ["ParallelizationTile", "ParallelizationThreads", "LoopInterchange", "Promotion", "Tiling"],
        ["ParallelizationTile", "Im2colLowering", "Tiling", "VectorizationSeq"],
        ["ParallelizationTile", "ParallelizationThreads", "Im2colLowering", "Tiling", "VectorizationSeq"],
        ["ParallelizationTile", "ParallelizationThreads", "Im2colLowering"],
        ["ParallelizationTile", "Tiling", "Im2colLowering"],
        ["ParallelizationTile", "Tiling", "Promotion", "Tiling"],
    ],
}

# claude --resume 5e5cb67b-7a49-4f81-b7fd-36d3c23cabd1 --dangerously-skip-permissions