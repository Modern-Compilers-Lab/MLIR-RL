from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v12.implementation.tiling import Tiling
from llm_action.src.actions.v12.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v12.implementation.promotion import Promotion
from llm_action.src.actions.v12.implementation.packing import Packing
from llm_action.src.actions.v12.implementation.vectorization import Vectorization
from llm_action.src.actions.v12.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v12.implementation.peeling import Peeling
from llm_action.src.actions.v12.implementation.parallelization import Parallelization
from llm_action.src.actions.v12.implementation.fusion import Fusion
from llm_action.src.actions.v12.implementation.loop_distribution import LoopDistribution

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Packing,
    Vectorization,
    LoopUnrolling,
    Peeling,
    Parallelization,
    Fusion,
    LoopDistribution,
]
