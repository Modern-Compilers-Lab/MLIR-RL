from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v22.implementation.tiling import Tiling
from llm_action.src.actions.v22.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v22.implementation.promotion import Promotion
from llm_action.src.actions.v22.implementation.vectorization import Vectorization
from llm_action.src.actions.v22.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v22.implementation.packing import Packing
from llm_action.src.actions.v22.implementation.parallelization import Parallelization
from llm_action.src.actions.v22.implementation.loop_peeling import LoopPeeling

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    LoopUnrolling,
    Packing,
    Parallelization,
    LoopPeeling,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {}
