from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v33.implementation.tiling import Tiling
from llm_action.src.actions.v33.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v33.implementation.packing import Packing
from llm_action.src.actions.v33.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v33.implementation.vectorization import Vectorization
from llm_action.src.actions.v33.implementation.promotion import Promotion
from llm_action.src.actions.v33.implementation.parallel_tiling import ParallelTiling
from llm_action.src.actions.v33.implementation.parallelization import Parallelization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Packing,
    Im2colLowering,
    Vectorization,
    Promotion,
    ParallelTiling,
    Parallelization,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # Packing only passes postcondition after Im2colLowering on conv_2d_nchw_fchw.
    # On raw conv or after Tiling/LoopInterchange/Promotion/ParallelTiling/Parallelization,
    # Packing consistently fails postcondition.
    "Packing": ["Im2colLowering"],
}

# Structural incompatibilities observed on conv_2d_nchw_fchw (empirical, 72 tests).
# If action X is chosen, the actions in the list become masked (cannot be applied next).
# Orientation: X -> [Y] means "choosing X masks Y".
ACTION_INCOMPATIBILITIES: dict[str, list[str]] = {
    # After Tiling, Vectorization and Packing both fail postcondition.
    "Tiling": ["Packing"],
    # After LoopInterchange (converts conv_2d to linalg.generic):
    # - Im2col fails precondition (needs conv_2d op, not generic)
    # - Vectorization/Packing fail postcondition
    "LoopInterchange": ["Im2colLowering", "Packing"],
    # After Im2colLowering (converts to batched-matmul-like generic):
    # - Im2col fails precondition (already lowered, no conv_2d op)
    # - Vectorization fails postcondition
    # - Promotion fails postcondition on im2col output
    # After Promotion (converts tensor -> memref):
    # - Im2col fails postcondition on memref form
    # - Vectorization/Packing fail postcondition
    "Promotion": ["Im2colLowering", "Packing"],
    # After ParallelTiling, Vectorization and Packing fail postcondition.
    "ParallelTiling": ["Packing"],
    # After Parallelization, Vectorization and Packing fail postcondition.
    "Parallelization": ["Packing"],
    # Vectorization always fails postcondition on conv_2d_nchw_fchw — every
    # subsequent action is moot since there is no valid output to continue from.
    # Packing similarly has no valid output on raw conv (only works after Im2col).
}
