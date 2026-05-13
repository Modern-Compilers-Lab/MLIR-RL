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

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v28")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            Values per slot: [0, 4, 8, 16, 32].

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
    Applies Promotion action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            Values per slot: [0, 4, 8, 16, 32].

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
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: packed_sizes (list[int]).
            Values per slot: [0, 2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Packing()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR Code.

    For conv2d ops, automatically applies Im2col lowering first to convert
    windowed access patterns into vectorizable projected-permutation maps,
    then vectorizes the resulting matmul-like operation.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]).
            Values per slot: [1, 2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Vectorization()
    if action.precondition(code, parameters):
        preprocessed = action.preprocess(code, parameters)
        transformed_code = action.implement(preprocessed, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Unrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]),
            unroll_factor (int). tile_sizes values: [0, 4, 8, 16, 32].
            unroll_factor values: [2, 3, 4].

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
def im2col_lowering_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Im2colLowering action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Empty dict (zero-parameter action).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Im2colLowering()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationTiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            Values per slot: [0, 4, 8, 16, 32].

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
    Applies ParallelizationDirect action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: num_threads (list[int]).
            Values per slot: [0, 2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelizationDirect()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def peeling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Peeling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]).
            Values per slot: [0, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Peeling()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
