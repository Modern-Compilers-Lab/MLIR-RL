from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v31.implementation.tiling import Tiling
from llm_action.src.actions.v31.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v31.implementation.vectorization import Vectorization
from llm_action.src.actions.v31.implementation.unrolling import Unrolling
from llm_action.src.actions.v31.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v31.implementation.parallelization_direct import ParallelizationDirect

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Vectorization,
    Unrolling,
    ParallelizationTiling,
    ParallelizationDirect,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # Empirically tested on pooling_nchw_max (11 kernels, 36/36 pairs):
    # ALL action pairs compose successfully (pre=T, post=T).
    # Vectorization generalizes pooling_nchw_max to linalg.generic, but
    # subsequent V and LI still pass both pre- and postconditions.
    # No block edges exist for this op family.
}
