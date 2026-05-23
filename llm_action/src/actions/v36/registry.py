from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v36.implementation.tiling import Tiling
from llm_action.src.actions.v36.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v36.implementation.promotion import Promotion
from llm_action.src.actions.v36.implementation.vectorization import Vectorization
from llm_action.src.actions.v36.implementation.unrolling import Unrolling
from llm_action.src.actions.v36.implementation.parallelization import Parallelization
from llm_action.src.actions.v36.implementation.parallelization_by_num_threads import ParallelizationByNumThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Unrolling,
    Parallelization,
    ParallelizationByNumThreads,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "LoopInterchange", "Vectorization", "Parallelization", "ParallelizationByNumThreads", "Unrolling"],
}
