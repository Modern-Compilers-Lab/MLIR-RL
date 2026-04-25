from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v15.implementation.tiling import Tiling
from llm_action.src.actions.v15.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v15.implementation.packing import Packing
from llm_action.src.actions.v15.implementation.promotion import Promotion
from llm_action.src.actions.v15.implementation.vectorization import Vectorization
from llm_action.src.actions.v15.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v15.implementation.padding import Padding
from llm_action.src.actions.v15.implementation.peeling import Peeling
from llm_action.src.actions.v15.implementation.parallelization import Parallelization
from llm_action.src.actions.v15.implementation.fusion import Fusion
from llm_action.src.actions.v15.implementation.loop_distribution import LoopDistribution
from llm_action.src.actions.v15.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v15.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v15.implementation.bufferization_strategy import BufferizationStrategy

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Packing,
    Promotion,
    Vectorization,
    LoopUnrolling,
    Padding,
    Peeling,
    Parallelization,
    Fusion,
    LoopDistribution,
    Im2colLowering,
    Canonicalization,
    BufferizationStrategy,
]
