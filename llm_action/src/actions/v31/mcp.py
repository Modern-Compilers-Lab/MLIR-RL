from fastmcp import FastMCP

from llm_action.src.actions.v31.implementation.tiling import Tiling
from llm_action.src.actions.v31.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v31.implementation.vectorization import Vectorization
from llm_action.src.actions.v31.implementation.unrolling import Unrolling
from llm_action.src.actions.v31.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v31.implementation.parallelization_direct import ParallelizationDirect

mcp = FastMCP("mlir-rl-action-space-v31")


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
            permutation is a reordering of loop indices, e.g. [1, 0, 2, 3, 4, 5].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopInterchange.precondition(code, parameters):
        transformed_code = LoopInterchange.implement(code, parameters)
        return True, transformed_code, LoopInterchange.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR Code.

    Tiles to SIMD-friendly sizes and generalizes pooling_nchw_max to
    linalg.generic, enabling LLVM auto-vectorization during lowering.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]).
            vector_sizes values from [1, 2, 4, 8, 16], one per loop dim.

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


if __name__ == "__main__":
    mcp.run()
