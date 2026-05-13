from fastmcp import FastMCP

from llm_action.src.actions.v30.implementation.tiling import Tiling
from llm_action.src.actions.v30.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v30.implementation.promotion import Promotion
from llm_action.src.actions.v30.implementation.vectorization import Vectorization
from llm_action.src.actions.v30.implementation.unrolling import Unrolling
from llm_action.src.actions.v30.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v30.implementation.parallelization_direct import ParallelizationDirect
from llm_action.src.actions.v30.implementation.image2col import Image2Col

mcp = FastMCP("mlir-rl-action-space-v30")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            tile_sizes values from [0, 4, 8, 16, 32], one per loop dim. 0 = no tiling.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Tiling.precondition(code, parameters):
        transformed_code = Tiling.implement(code, parameters)
        return True, transformed_code, Tiling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_interchange_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopInterchange action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: permutation (list[int]).
            permutation is a reordering of loop indices, e.g. [1, 0, 2, 3, 4, 5, 6].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopInterchange.precondition(code, parameters):
        transformed_code = LoopInterchange.implement(code, parameters)
        return True, transformed_code, LoopInterchange.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: operands_to_promote (list[int]).
            operands_to_promote selects which operands to copy into contiguous buffers.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Promotion.precondition(code, parameters):
        transformed_code = Promotion.implement(code, parameters)
        return True, transformed_code, Promotion.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]).
            vector_sizes values from [1, 2, 4, 8, 16], one per loop dim.
            For conv2d, internally converts to img2col before vectorizing.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Vectorization.precondition(code, parameters):
        transformed_code = Vectorization.implement(code, parameters)
        return True, transformed_code, Vectorization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Unrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            tile_sizes values from [0, 2, 4, 8, 16]; first non-zero entry selects
            the loop dimension and unroll factor.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Unrolling.precondition(code, parameters):
        transformed_code = Unrolling.implement(code, parameters)
        return True, transformed_code, Unrolling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationTiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            tile_sizes values from [0, 4, 8, 16, 32], per loop dim for parallel distribution.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if ParallelizationTiling.precondition(code, parameters):
        transformed_code = ParallelizationTiling.implement(code, parameters)
        return True, transformed_code, ParallelizationTiling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_direct_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationDirect action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: num_threads (list[int]).
            num_threads values from [0, 2, 4, 7, 14], per loop dim for direct thread assignment.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if ParallelizationDirect.precondition(code, parameters):
        transformed_code = ParallelizationDirect.implement(code, parameters)
        return True, transformed_code, ParallelizationDirect.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def image2col_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Image2Col action on MLIR Code.

    Converts linalg.conv_2d_nchw_fchw to a matmul-like contraction
    (linalg.generic) via img2col transformation. The resulting operation
    has 4 loops instead of 7: [batch, M, N, K] where M=OH*OW, N=F, K=C*KH*KW.

    Args:
        code (str): The MLIR code containing a tagged conv2d operation.
        parameters (dict): Parameters for the action (empty dict; parameterless).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Image2Col.precondition(code, parameters):
        transformed_code = Image2Col.implement(code, parameters)
        return True, transformed_code, Image2Col.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
