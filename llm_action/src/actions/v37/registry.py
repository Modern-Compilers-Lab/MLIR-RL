from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v37.implementation.tiling import Tiling
from llm_action.src.actions.v37.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v37.implementation.vectorization import Vectorization
from llm_action.src.actions.v37.implementation.unrolling import Unrolling
from llm_action.src.actions.v37.implementation.parallelization import Parallelization
from llm_action.src.actions.v37.implementation.parallelization_by_num_threads import ParallelizationByNumThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Vectorization,
    Unrolling,
    Parallelization,
    ParallelizationByNumThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "LoopInterchange", "Vectorization", "Parallelization", "ParallelizationByNumThreads", "Unrolling"],
}
