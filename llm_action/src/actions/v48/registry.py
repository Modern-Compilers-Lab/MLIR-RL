from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v48.implementation.tiling import Tiling
from llm_action.src.actions.v48.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v48.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v48.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v48.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v48.implementation.parallelization_threads import ParallelizationThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    VectorizationSequential,
    VectorizationParallel,
    ParallelizationTiling,
    ParallelizationThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "VectorizationSequential": ["Tiling", "LoopInterchange", "VectorizationParallel", "ParallelizationTiling", "ParallelizationThreads"],
    "VectorizationParallel": ["Tiling", "LoopInterchange", "VectorizationSequential", "ParallelizationTiling", "ParallelizationThreads"],
}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "matmul": [
        ["VectorizationParallel"],
        ["ParallelizationThreads", "VectorizationSequential"],
        ["ParallelizationTiling", "VectorizationSequential"],
        ["ParallelizationTiling", "Tiling", "VectorizationSequential"],
        ["ParallelizationTiling", "LoopInterchange", "VectorizationParallel"],
    ],
}
