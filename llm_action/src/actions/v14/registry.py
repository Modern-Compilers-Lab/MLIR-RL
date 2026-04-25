from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v14.implementation.tiling import Tiling
from llm_action.src.actions.v14.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v14.implementation.packing import Packing
from llm_action.src.actions.v14.implementation.promotion import Promotion
from llm_action.src.actions.v14.implementation.vectorization import Vectorization
from llm_action.src.actions.v14.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v14.implementation.peeling import Peeling
from llm_action.src.actions.v14.implementation.padding import Padding
from llm_action.src.actions.v14.implementation.parallelization import Parallelization
from llm_action.src.actions.v14.implementation.fusion import Fusion
from llm_action.src.actions.v14.implementation.loop_distribution import LoopDistribution
from llm_action.src.actions.v14.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v14.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v14.implementation.bufferization_strategy import BufferizationStrategy

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Packing,
    Promotion,
    Vectorization,
    LoopUnrolling,
    Peeling,
    Padding,
    Parallelization,
    Fusion,
    LoopDistribution,
    Canonicalization,
    Im2colLowering,
    BufferizationStrategy,
]
