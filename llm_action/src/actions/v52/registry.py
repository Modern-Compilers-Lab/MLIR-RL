from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v52.implementation.tiling import Tiling
from llm_action.src.actions.v52.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v52.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v52.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v52.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v52.implementation.parallelization_threads import ParallelizationThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    VectorizationSequential,
    VectorizationParallel,
    ParallelizationTiling,
    ParallelizationThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "VectorizationSequential": [
        "Tiling",
        "LoopInterchange",
        "VectorizationSequential",
        "VectorizationParallel",
        "ParallelizationTiling",
        "ParallelizationThreads",
    ],
    "VectorizationParallel": [
        "Tiling",
        "LoopInterchange",
        "VectorizationSequential",
        "VectorizationParallel",
        "ParallelizationTiling",
        "ParallelizationThreads",
    ],
}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "relu": [
        ["ParallelizationThreads"],
        ["VectorizationParallel"],
        ["ParallelizationThreads", "VectorizationParallel"],
        ["Tiling", "VectorizationParallel"],
        ["LoopInterchange", "VectorizationParallel"],
        ["ParallelizationThreads", "Tiling", "VectorizationParallel"],
    ],
}