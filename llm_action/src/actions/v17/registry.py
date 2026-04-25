from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v17.implementation.tiling import Tiling
from llm_action.src.actions.v17.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v17.implementation.promotion import Promotion
from llm_action.src.actions.v17.implementation.vectorization import Vectorization
from llm_action.src.actions.v17.implementation.unrolling import Unrolling
from llm_action.src.actions.v17.implementation.parallelization import Parallelization
from llm_action.src.actions.v17.implementation.kernel_lowering import KernelLowering

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Unrolling,
    Parallelization,
    KernelLowering,
]
