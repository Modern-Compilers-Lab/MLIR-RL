from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v28.implementation.tiling import Tiling
from llm_action.src.actions.v28.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v28.implementation.promotion import Promotion
from llm_action.src.actions.v28.implementation.packing import Packing
from llm_action.src.actions.v28.implementation.vectorization import Vectorization
from llm_action.src.actions.v28.implementation.unrolling import Unrolling
from llm_action.src.actions.v28.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v28.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v28.implementation.parallelization_direct import ParallelizationDirect
from llm_action.src.actions.v28.implementation.peeling import Peeling

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Packing,
    Vectorization,
    Unrolling,
    Im2colLowering,
    ParallelizationTiling,
    ParallelizationDirect,
    Peeling,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # Empirically derived from conv_2d_nchw_fchw exploration (v28, paper_conv2d train set).
    # Format: "applied_action": ["actions_masked_after_it", ...]
    # When action X is executed, every action in ACTION_DEPENDENCIES[X] is masked
    # (forbidden from selection in subsequent steps).
    #
    # Note: self-blocks (X blocks X) are skipped by action_registry.py (line 38-39).
    #
    # Vectorization now works on conv2d via automatic Im2col preprocessing
    # (converts conv to matmul-like generic, then vectorizes). It is terminal:
    # consumes the linalg op and replaces it with vector ops.
    #
    # Packing (Post=F) and Peeling (Post=F) remain structurally incompatible
    # with conv_2d_nchw_fchw.
    #
    "Tiling": ["Packing", "Peeling"],
    "LoopInterchange": ["Vectorization", "Packing", "Peeling", "Im2colLowering"],
    "Promotion": [  # Terminal: converts to memref, blocks all subsequent actions
        "Tiling", "LoopInterchange", "Promotion", "Packing",
        "Vectorization", "Unrolling", "Im2colLowering",
        "ParallelizationTiling", "ParallelizationDirect", "Peeling",
    ],
    "Packing": ["Peeling"],  # No-op (Post=F); cross-mask Peeling
    "Vectorization": [  # Terminal: consumes linalg op, produces vector ops
        "Tiling", "LoopInterchange", "Promotion", "Packing",
        "Unrolling", "Im2colLowering",
        "ParallelizationTiling", "ParallelizationDirect", "Peeling",
    ],
    "Unrolling": ["Packing", "Peeling"],
    "Im2colLowering": ["Packing", "Peeling", "LoopInterchange", "Im2colLowering"],
    "ParallelizationTiling": ["Packing", "Peeling"],
    "ParallelizationDirect": ["Packing", "Peeling"],
    "Peeling": ["Packing"],  # No-op (Post=F); cross-mask Packing
}

# Best schedules discovered (conv_2d_nchw_fchw, paper_conv2d/train):
#
# 1x1 convolutions (KH=KW=1):
#   K1 (256x128x28x28 -> 256x512x28x28):
#     Im2col -> PT[8,16] -> T[0,0,16,16]: 241.65ms (75.5x vs base, 0.17x PyTorch)
#     PT[8,16] -> V[4,8,16,8]: 364.15ms (50.1x vs base, 0.11x PyTorch)
#   K2 (256x512x28x28 -> 256x128x28x28):
#     PT[8,16] -> T[0,0,4,4,8,0,0]: 399.93ms (48.0x vs base, 0.13x PyTorch)
#     Im2col -> PT[8,16] -> T[0,0,16,16]: 425.97ms (45.0x vs base, 0.12x PyTorch)
#
# 3x3 convolutions (KH=KW=3):
#   K3 (256x64x56x56 -> 256x64x54x54):
#     PT[8,8] -> T[0,0,4,4,8,0,0]: 1089.56ms (0.17x PyTorch)
#     Im2col -> PT -> T: 6300.91ms (HARMFUL - 9x expansion)
#
# Strategy:
#   - 1x1 conv: Im2col -> PT -> T (best) or PT -> V (vectorized alternative)
#   - 3x3+ conv: PT -> T only (skip Im2col - expansion factor KH*KW too large)
#   - Vectorization: works via auto-Im2col, but scalar tiled code often faster
