from llm_action.src.actions.base import ActionBase
from llm_action.src.actions.v4.implementation.tiling import Tiling
from llm_action.src.actions.v4.implementation.multi_level_tiling import MultiLevelTiling
from llm_action.src.actions.v4.implementation.promotion import Promotion
from llm_action.src.actions.v4.implementation.packing import Packing
from llm_action.src.actions.v4.implementation.vectorization import Vectorization
from llm_action.src.actions.v4.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v4.implementation.peeling import Peeling
from llm_action.src.actions.v4.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v4.implementation.parallelization import Parallelization
from llm_action.src.actions.v4.implementation.loop_fusion import LoopFusion
from llm_action.src.actions.v4.implementation.loop_distribution import LoopDistribution
from llm_action.src.actions.v4.implementation.decomposition import Decomposition
from llm_action.src.actions.v4.implementation.padding import Padding
from llm_action.src.actions.v4.implementation.generalization import Generalization
from llm_action.src.actions.v4.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v4.implementation.unroll_and_jam import UnrollAndJam
from llm_action.src.actions.v4.implementation.scalar_replacement import ScalarReplacement
from llm_action.src.actions.v4.implementation.loop_coalescing import LoopCoalescing

ACTION_CLASSES: list[type[ActionBase]] = [
    Tiling, MultiLevelTiling, Promotion, Packing,
    Vectorization, LoopInterchange, Peeling, LoopUnrolling,
    Parallelization, LoopFusion, LoopDistribution,
    Decomposition, Padding, Generalization, Canonicalization,
    UnrollAndJam, ScalarReplacement, LoopCoalescing,
]
