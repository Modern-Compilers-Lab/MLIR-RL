from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v53.implementation.tiling import Tiling
from llm_action.src.actions.v53.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v53.implementation.promotion import Promotion
from llm_action.src.actions.v53.implementation.sequential_vectorization import SequentialVectorization
from llm_action.src.actions.v53.implementation.parallel_vectorization import ParallelVectorization
from llm_action.src.actions.v53.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v53.implementation.tiling_parallelization import TilingParallelization
from llm_action.src.actions.v53.implementation.thread_parallelization import ThreadParallelization

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    SequentialVectorization,
    ParallelVectorization,
    Im2colLowering,
    TilingParallelization,
    ThreadParallelization,
]

# Cross-kernel denylist: actions whose precondition ALWAYS fails for a family,
# or whose postcondition ALWAYS fails regardless of IR state.
# Key: kernel family name. Value: list of action class names that are never valid.
ACTION_DEPENDENCIES: dict[str, list[str]] = {
    "matmul": ["Im2colLowering"],
    "conv2d": [],
    "pooling": ["Im2colLowering", "SequentialVectorization", "ParallelVectorization"],
    "add": ["Im2colLowering", "LoopInterchange"],
    "relu": ["Im2colLowering", "LoopInterchange"],
}

# Per-family allowlist of empirically validated schedule paths (action-sequence skeletons).
# Each path is a list of action class names applied in order.
# Vectorization actions (SequentialVectorization, ParallelVectorization) are TERMINAL:
# no further actions can compose after them (tag moves from linalg op to scf loop).
# Pruned to top 5-7 per family by average speedup over base MLIR across 3 representative
# kernels (small/medium/large) per family. Benchmarked on Intel Xeon E5-2680 v4 (28 cores).
SCHEDULE_GRAPH: dict[str, list[list[str]]] = {
    "matmul": [
        # 193x avg speedup — best overall, two-level parallelism + vectorization
        ["TilingParallelization", "ParallelVectorization"],
        # 186x avg — single-step, nearly as good as TilingPar+ParVec
        ["ParallelVectorization"],
        # 155x avg — loop reorder improves vectorization access pattern
        ["LoopInterchange", "ParallelVectorization"],
        # 92x avg — sequential vectorization within parallel tiles
        ["TilingParallelization", "SequentialVectorization"],
        # 55x avg — two-level tiling for cache locality (non-terminal)
        ["TilingParallelization", "Tiling"],
        # Beats TilingPar+ParVec by 8.3% on 5 diverse kernels — cache-level
        # tiling between parallel tiles and vectorization improves locality,
        # especially for reduction-heavy shapes (+97% on 128x1024x128)
        ["TilingParallelization", "Tiling", "ParallelVectorization"],
        # 30x avg — simple parallelization baseline (non-terminal)
        ["TilingParallelization"],
    ],
    "conv2d": [
        # 30x avg — parallel tiles + reduction tiling for cache locality
        ["TilingParallelization", "Tiling"],
        # 28.4x avg — loop reorder + parallel tiles + tiling (diverse LI prefix)
        ["LoopInterchange", "TilingParallelization", "Tiling"],
        # 28.4x avg — loop reorder + parallel tiles (diverse LI prefix)
        ["LoopInterchange", "TilingParallelization"],
        # 22x avg — single-step parallelization, strong and simple
        ["TilingParallelization"],
        # 16x avg — consistent thread-level parallelism
        ["ThreadParallelization"],
        # 14x avg — im2col converts conv to matmul-like, then parallelize
        ["Im2colLowering", "TilingParallelization"],
        # 8.1x avg — promotion copies operands to fast buffers, then parallelize
        # (diverse Promotion prefix; limited by copy overhead on large tensors)
        ["Promotion", "ThreadParallelization"],
        # 3x avg — im2col + sequential tiling (marginal, large-kernel benefit)
        ["Im2colLowering", "Tiling"],
    ],
    "pooling": [
        # 8.7x avg, 2.9x vs torch — two-level parallelism (best overall)
        ["TilingParallelization", "ThreadParallelization"],
        # 8.7x avg, 2.9x vs torch — single-step parallel tiling
        ["TilingParallelization"],
        # 8.4x avg, 2.7x vs torch — simple thread parallelism
        ["ThreadParallelization"],
        # 7.4x avg, 2.4x vs torch — parallel tiles + sequential tiling
        ["TilingParallelization", "Tiling"],
        # 7.1x avg — thread par + tiling + parallel tiling (diverse ThreadPar prefix)
        ["ThreadParallelization", "Tiling", "TilingParallelization"],
        # 7.1x avg — thread par + parallel tiling (diverse ThreadPar prefix)
        ["ThreadParallelization", "TilingParallelization"],
    ],
    "add": [
        # 5.7x avg, 1.1x vs torch — beats PyTorch on medium/large
        ["ThreadParallelization"],
        # 4.8x avg, 0.9x vs torch — multi-dim parallel tiling
        ["TilingParallelization"],
        # 2.8x avg — vectorization with implicit parallelism
        ["ParallelVectorization"],
        # 2.5x avg — parallel tiles then sequential vectorization
        ["TilingParallelization", "SequentialVectorization"],
        # 1.6x avg — parallel tiles then parallel vectorization
        ["TilingParallelization", "ParallelVectorization"],
    ],
    "relu": [
        # 4.7x avg, 1.2x vs torch — beats PyTorch on medium/large
        ["ThreadParallelization"],
        # 3.7x avg, 1.1x vs torch — multi-dim parallel tiling
        ["TilingParallelization"],
        # 2.8x avg — vectorization with implicit parallelism
        ["ParallelVectorization"],
        # 2.8x avg — parallel tiles then parallel vectorization
        ["TilingParallelization", "ParallelVectorization"],
        # 2.7x avg — parallel tiles then sequential vectorization
        ["TilingParallelization", "SequentialVectorization"],
    ],
}