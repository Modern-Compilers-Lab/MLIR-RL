from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v13.implementation.tiling import Tiling
from llm_action.src.actions.v13.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v13.implementation.packing import Packing
from llm_action.src.actions.v13.implementation.promotion import Promotion
from llm_action.src.actions.v13.implementation.vectorization import Vectorization
from llm_action.src.actions.v13.implementation.unrolling import Unrolling
from llm_action.src.actions.v13.implementation.peeling import Peeling
from llm_action.src.actions.v13.implementation.padding import Padding
from llm_action.src.actions.v13.implementation.parallelization import Parallelization
from llm_action.src.actions.v13.implementation.fusion import Fusion
from llm_action.src.actions.v13.implementation.canonicalization import Canonicalization

# ACTION_CLASSES: list[type[ActionBase]] = [
#     Tiling,
#     LoopInterchange,
#     Packing,
#     Promotion,
#     Vectorization,
#     Unrolling,
#     Peeling,
#     Padding,
#     Parallelization,
#     Fusion
# ]

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Vectorization,
    Parallelization,
    Fusion
]
