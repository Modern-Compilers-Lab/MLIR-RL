from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v30.implementation.tiling import Tiling
from llm_action.src.actions.v30.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v30.implementation.promotion import Promotion
from llm_action.src.actions.v30.implementation.vectorization import Vectorization
from llm_action.src.actions.v30.implementation.unrolling import Unrolling
from llm_action.src.actions.v30.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v30.implementation.parallelization_direct import ParallelizationDirect
from llm_action.src.actions.v30.implementation.image2col import Image2Col

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling,
    LoopInterchange,
    Promotion,
    Vectorization,
    Unrolling,
    ParallelizationTiling,
    ParallelizationDirect,
    Image2Col,
]

ACTION_DEPENDENCIES: dict[str, list[str]] = {
    # Block edges: if key action X has executed, value actions become unavailable.
    # Empirically verified on matmul and conv2d (dataset_conv2d, v30, 2026-05-11).
    # Include only edges where Y after X failed pre or post on EVERY tested kernel.
    #
    # Vectorization: replaces tagged linalg op with vector ops (via img2col+vectorize
    # for conv2d), destroying the tag. All subsequent actions fail precondition.
    "Vectorization": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Vectorization",
        "Unrolling",
        "ParallelizationTiling",
        "ParallelizationDirect",
        "Image2Col",
    ],
    # Unrolling: replaces tagged op with unrolled scalar/loop form, tag is lost.
    "Unrolling": [
        "Tiling",
        "LoopInterchange",
        "Promotion",
        "Vectorization",
        "Unrolling",
        "ParallelizationTiling",
        "ParallelizationDirect",
        "Image2Col",
    ],
    # No block edges found for: Tiling, LoopInterchange, Promotion,
    # ParallelizationTiling, ParallelizationDirect, Image2Col.
    # All tested pairwise compositions from PT->{T,V,I2C,P,U,LI} and
    # I2C->{T,PT,V} passed pre=T, post=T on conv2d K1.
}
