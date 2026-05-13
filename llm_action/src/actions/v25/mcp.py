from fastmcp import FastMCP

from llm_action.src.actions.v25.implementation.tiling import Tiling
from llm_action.src.actions.v25.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v25.implementation.promotion import Promotion
from llm_action.src.actions.v25.implementation.packing import Packing
from llm_action.src.actions.v25.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v25.implementation.vectorization import Vectorization
from llm_action.src.actions.v25.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v25.implementation.loop_peeling import LoopPeeling
from llm_action.src.actions.v25.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v25.implementation.tiling_based_parallelization import TilingBasedParallelization
from llm_action.src.actions.v25.implementation.thread_count_parallelization import ThreadCountParallelization
from llm_action.src.actions.v25.implementation.split_reduction import SplitReduction

mcp = FastMCP("rl-action-v25")


@mcp.tool()
def tiling_tool(code: str, tile_sizes: list[int]) -> str:
    """
    Tile a tagged linalg operation using tile_using_for.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        tile_sizes: Tile size per loop dimension (0 = skip).
    Returns:
        Transformed MLIR code.
    """
    params = {"tile_sizes": tile_sizes}
    if not Tiling.precondition(code, params):
        return code
    result = Tiling.implement(code, params)
    if Tiling.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def loop_interchange_tool(code: str, permutation: list[int]) -> str:
    """
    Reorder loop dimensions via generalize + interchange.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        permutation: Non-identity permutation vector for loop dimensions.
    Returns:
        Transformed MLIR code.
    """
    params = {"permutation": permutation}
    if not LoopInterchange.precondition(code, params):
        return code
    result = LoopInterchange.implement(code, params)
    if LoopInterchange.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def promotion_tool(code: str, operands_to_promote: list[int]) -> str:
    """
    Promote tiled operands into contiguous temporary buffers.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        operands_to_promote: List of operand indices to promote.
    Returns:
        Transformed MLIR code.
    """
    params = {"operands_to_promote": operands_to_promote}
    if not Promotion.precondition(code, params):
        return code
    result = Promotion.implement(code, params)
    if Promotion.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def packing_tool(code: str, packed_sizes: list[int]) -> str:
    """
    Pack a linalg operation by introducing inner blocking dimensions.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        packed_sizes: Inner block size per dimension (0 = skip).
    Returns:
        Transformed MLIR code.
    """
    params = {"packed_sizes": packed_sizes}
    if not Packing.precondition(code, params):
        return code
    result = Packing.implement(code, params)
    if Packing.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def im2col_lowering_tool(code: str) -> str:
    """
    Convert a conv2d operation into img2col + matmul.
    Args:
        code: MLIR code containing a conv2d operation tagged with "operation_0".
    Returns:
        Transformed MLIR code.
    """
    params = {}
    if not Im2colLowering.precondition(code, params):
        return code
    result = Im2colLowering.implement(code, params)
    if Im2colLowering.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def vectorization_tool(code: str, vector_sizes: list[int]) -> str:
    """
    Vectorize a linalg operation by tiling and mapping to SIMD vectors.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        vector_sizes: SIMD width per loop dimension.
    Returns:
        Transformed MLIR code.
    """
    params = {"vector_sizes": vector_sizes}
    if not Vectorization.precondition(code, params):
        return code
    result = Vectorization.implement(code, params)
    if Vectorization.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def loop_unrolling_tool(code: str, unroll_factor: int) -> str:
    """
    Unroll the innermost loop of a tagged operation.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        unroll_factor: Number of iterations to unroll (2, 4, 8, 16).
    Returns:
        Transformed MLIR code.
    """
    params = {"unroll_factor": unroll_factor}
    if not LoopUnrolling.precondition(code, params):
        return code
    result = LoopUnrolling.implement(code, params)
    if LoopUnrolling.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def loop_peeling_tool(code: str, tile_size: int) -> str:
    """
    Peel remainder iterations into a cleanup loop.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        tile_size: Tile size for the reduction dimension before peeling.
    Returns:
        Transformed MLIR code.
    """
    params = {"tile_size": tile_size}
    if not LoopPeeling.precondition(code, params):
        return code
    result = LoopPeeling.implement(code, params)
    if LoopPeeling.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def canonicalization_tool(code: str) -> str:
    """
    Apply canonicalization to simplify and normalize IR.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
    Returns:
        Transformed MLIR code.
    """
    params = {}
    if not Canonicalization.precondition(code, params):
        return code
    result = Canonicalization.implement(code, params)
    if Canonicalization.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def tiling_based_parallelization_tool(code: str, tile_sizes: list[int]) -> str:
    """
    Tile and distribute parallel dimensions via scf.forall (tile_sizes).
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        tile_sizes: Tile sizes for parallel loop dimensions.
    Returns:
        Transformed MLIR code.
    """
    params = {"tile_sizes": tile_sizes}
    if not TilingBasedParallelization.precondition(code, params):
        return code
    result = TilingBasedParallelization.implement(code, params)
    if TilingBasedParallelization.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def thread_count_parallelization_tool(code: str, num_threads: int) -> str:
    """
    Partition iterations across a fixed number of threads via scf.forall.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        num_threads: Number of worker threads.
    Returns:
        Transformed MLIR code.
    """
    params = {"num_threads": num_threads}
    if not ThreadCountParallelization.precondition(code, params):
        return code
    result = ThreadCountParallelization.implement(code, params)
    if ThreadCountParallelization.postcondition(code, result, params):
        return result
    return code


@mcp.tool()
def split_reduction_tool(code: str, split_factor: int) -> str:
    """
    Split a reduction loop into independent partial reductions.
    Args:
        code: MLIR code containing an operation tagged with "operation_0".
        split_factor: Number of partial reductions.
    Returns:
        Transformed MLIR code.
    """
    params = {"split_factor": split_factor}
    if not SplitReduction.precondition(code, params):
        return code
    result = SplitReduction.implement(code, params)
    if SplitReduction.postcondition(code, result, params):
        return result
    return code


if __name__ == "__main__":
    mcp.run()
