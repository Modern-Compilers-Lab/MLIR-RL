from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v10.implementation.tiling import Tiling
from llm_action.src.actions.v10.implementation.packing import Packing
from llm_action.src.actions.v10.implementation.vectorization import Vectorization
from llm_action.src.actions.v10.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v10.implementation.parallelization import Parallelization
from llm_action.src.actions.v10.implementation.loop_unrolling import LoopUnrolling

# ACTION_CLASSES: list[type[ActionBase]] = [
#     Tiling,
#     Packing,
#     Vectorization,
#     LoopInterchange,
#     Parallelization,
#     LoopUnrolling
# ]

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    Vectorization,
    LoopInterchange,
    Parallelization
]
