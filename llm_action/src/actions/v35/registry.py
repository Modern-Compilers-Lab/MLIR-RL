from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v35.implementation.tiling import Tiling
from llm_action.src.actions.v35.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v35.implementation.vectorization import Vectorization
from llm_action.src.actions.v35.implementation.parallelization import Parallelization
from llm_action.src.actions.v35.implementation.parallelization_by_num_threads import ParallelizationByNumThreads
from llm_action.src.actions.v35.implementation.unrolling import Unrolling

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Vectorization,
    Parallelization,
    ParallelizationByNumThreads,
    Unrolling,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "LoopInterchange", "Vectorization", "Parallelization", "ParallelizationByNumThreads", "Unrolling"],
}
