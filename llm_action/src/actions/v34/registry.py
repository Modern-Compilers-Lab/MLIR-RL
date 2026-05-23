from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v34.implementation.tiling import Tiling
from llm_action.src.actions.v34.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v34.implementation.promotion import Promotion
from llm_action.src.actions.v34.implementation.vectorization import Vectorization
from llm_action.src.actions.v34.implementation.unrolling import Unrolling
from llm_action.src.actions.v34.implementation.parallelization import Parallelization
from llm_action.src.actions.v34.implementation.parallelization_by_num_threads import ParallelizationByNumThreads

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Unrolling,
    Parallelization,
    ParallelizationByNumThreads,
]

# Confirmed by exhaustive pairwise testing on:
#   - pooling_nchw_max_256_128_28_28_1_14_14 (49 pairs, all tested empirically)
# Vectorization: terminal action — blocks all follow-up actions (rewrites linalg to vector ops)
# Promotion: blocks only Vectorization (bufferizes tensor->memref; Vec requires tensor-level linalg)
# All other pairs compose freely (T, LI, P, U, Pa, PBT in any order).
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "Vectorization": ["Tiling", "LoopInterchange", "Promotion", "Vectorization", "Unrolling", "Parallelization", "ParallelizationByNumThreads"],
    "Promotion": ["Vectorization"],
}
