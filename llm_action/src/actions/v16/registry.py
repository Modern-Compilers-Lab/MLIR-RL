from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v16.implementation.tiling import Tiling
from llm_action.src.actions.v16.implementation.packing import Packing
from llm_action.src.actions.v16.implementation.promotion import Promotion
from llm_action.src.actions.v16.implementation.vectorization import Vectorization
from llm_action.src.actions.v16.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v16.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v16.implementation.parallelization import Parallelization
from llm_action.src.actions.v16.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v16.implementation.peeling import Peeling

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    Packing,
    Promotion,
    Vectorization,
    LoopInterchange,
    LoopUnrolling,
    Parallelization,
    Im2colLowering,
    Peeling,
]
