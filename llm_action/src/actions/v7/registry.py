from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v7.implementation.tiling import Tiling
from llm_action.src.actions.v7.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v7.implementation.vectorization import Vectorization
from llm_action.src.actions.v7.implementation.parallelization import Parallelization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Vectorization,
    Parallelization,
]
