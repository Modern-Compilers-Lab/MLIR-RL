from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v8.implementation.tiling import Tiling
from llm_action.src.actions.v8.implementation.packing import Packing
from llm_action.src.actions.v8.implementation.vectorization import Vectorization
from llm_action.src.actions.v8.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v8.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v8.implementation.parallelization import Parallelization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    Packing,
    Vectorization,
    LoopUnrolling,
    LoopInterchange,
    Parallelization,
]
