from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v46.implementation.tiling import Tiling
from llm_action.src.actions.v46.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v46.implementation.packing import Packing
from llm_action.src.actions.v46.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v46.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v46.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v46.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v46.implementation.parallelization_threads import ParallelizationThreads
from llm_action.src.actions.v46.implementation.promotion import Promotion

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Packing,
    VectorizationSequential,
    VectorizationParallel,
    LoopUnrolling,
    ParallelizationTiling,
    ParallelizationThreads,
    Promotion,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {}

SCHEDULE_GRAPH: dict[str, list[list[str]]] = {}
