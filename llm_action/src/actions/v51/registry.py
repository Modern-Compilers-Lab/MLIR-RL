from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v51.implementation.tiling import Tiling
from llm_action.src.actions.v51.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v51.implementation.vectorization_seq import VectorizationSeq
from llm_action.src.actions.v51.implementation.vectorization_par import VectorizationPar
from llm_action.src.actions.v51.implementation.parallelization_tile import ParallelizationTile
from llm_action.src.actions.v51.implementation.parallelization_threads import ParallelizationThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    VectorizationSeq,
    VectorizationPar,
    ParallelizationTile,
    ParallelizationThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "VectorizationSeq": ["Tiling", "LoopInterchange", "VectorizationSeq", "VectorizationPar", "ParallelizationTile", "ParallelizationThreads"],
    "VectorizationPar": ["Tiling", "LoopInterchange", "VectorizationSeq", "VectorizationPar", "ParallelizationTile", "ParallelizationThreads"],
}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "add": [
        ["VectorizationPar"],
        ["ParallelizationThreads", "VectorizationPar"],
        ["ParallelizationTile", "VectorizationPar"],
        ["ParallelizationTile", "VectorizationSeq"],
        ["ParallelizationTile", "ParallelizationThreads", "VectorizationPar"],
    ],
}
