from llm_action.src.actions.v27.implementation.tiling import Tiling
from llm_action.src.actions.v27.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v27.implementation.promotion import Promotion
from llm_action.src.actions.v27.implementation.vectorization import Vectorization
from llm_action.src.actions.v27.implementation.unrolling import Unrolling
from llm_action.src.actions.v27.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v27.implementation.parallelization_direct import ParallelizationDirect

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v27")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            tile_sizes values: [0, 4, 8, 16, 32] per dimension (0 = skip).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Tiling()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_interchange_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopInterchange action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: permutation (list[int]).
            permutation: a valid non-identity permutation of loop indices.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = LoopInterchange()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code (tile + bufferize + promote).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            tile_sizes values: [16, 32, 64, 128] per dimension.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Promotion()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR Code (tile to vector sizes + vectorize).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]).
            vector_sizes values: [2, 4, 8, 16] per dimension (SIMD widths).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Vectorization()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Unrolling action on MLIR Code (tile + unroll innermost loop).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]),
            unroll_factor (int). tile_sizes values: [4, 8, 16, 32].
            unroll_factor values: [2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Unrolling()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationTiling action on MLIR Code (tile_using_forall with tile_sizes).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            tile_sizes values: [0, 16, 32, 64, 128] per dimension (0 = skip).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelizationTiling()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_direct_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationDirect action on MLIR Code (tile_using_forall with num_threads).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: num_threads (list[int]).
            num_threads values: [0, 2, 4, 8, 16] per dimension (0 = skip).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelizationDirect()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
