from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v18.implementation.tiling import Tiling
from llm_action.src.actions.v18.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v18.implementation.promotion import Promotion
from llm_action.src.actions.v18.implementation.vectorization import Vectorization
from llm_action.src.actions.v18.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v18.implementation.parallelization import Parallelization
from llm_action.src.actions.v18.implementation.packing import Packing

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    LoopUnrolling,
    Parallelization,
    Packing,
]

# Vectorization replaces the tagged linalg op with vector.transfer_read/write + arith ops;
# no subsequent action can locate the tagged operation.
# Promotion converts tensor->memref but preserves the tagged linalg op; all actions still compose.
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "Parallelization", "LoopUnrolling", "LoopInterchange", "Packing", "Promotion"],
}
