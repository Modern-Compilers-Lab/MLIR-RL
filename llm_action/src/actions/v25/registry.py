from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v25.implementation.tiling import Tiling
from llm_action.src.actions.v25.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v25.implementation.promotion import Promotion
from llm_action.src.actions.v25.implementation.packing import Packing
from llm_action.src.actions.v25.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v25.implementation.vectorization import Vectorization
from llm_action.src.actions.v25.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v25.implementation.loop_peeling import LoopPeeling
from llm_action.src.actions.v25.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v25.implementation.tiling_based_parallelization import TilingBasedParallelization
from llm_action.src.actions.v25.implementation.thread_count_parallelization import ThreadCountParallelization
from llm_action.src.actions.v25.implementation.split_reduction import SplitReduction

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Packing,
    Im2colLowering,
    Vectorization,
    LoopUnrolling,
    LoopPeeling,
    Canonicalization,
    TilingBasedParallelization,
    ThreadCountParallelization,
    SplitReduction,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {}
