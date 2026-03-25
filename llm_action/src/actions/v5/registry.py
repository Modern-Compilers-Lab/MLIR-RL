from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v5.implementation.tiling import Tiling
from llm_action.src.actions.v5.implementation.packing import Packing
from llm_action.src.actions.v5.implementation.vectorization import Vectorization
from llm_action.src.actions.v5.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v5.implementation.peeling import Peeling
from llm_action.src.actions.v5.implementation.unrolling import Unrolling
from llm_action.src.actions.v5.implementation.parallelization import Parallelization
from llm_action.src.actions.v5.implementation.fusion import Fusion
from llm_action.src.actions.v5.implementation.canonicalization import Canonicalization

# ACTION_CLASSES: list[type[ActionBase]] = [
#     Tiling, Packing, Vectorization, LoopInterchange,
#     Peeling, Unrolling, Parallelization, Fusion, Canonicalization,
# ]

# Optimize selection
ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling, Packing, Vectorization, LoopInterchange,
    Peeling, Unrolling, Parallelization, Fusion
]
