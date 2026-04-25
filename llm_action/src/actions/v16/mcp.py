from llm_action.src.actions.v16.implementation.tiling import Tiling
from llm_action.src.actions.v16.implementation.packing import Packing
from llm_action.src.actions.v16.implementation.promotion import Promotion
from llm_action.src.actions.v16.implementation.vectorization import Vectorization
from llm_action.src.actions.v16.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v16.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v16.implementation.parallelization import Parallelization
from llm_action.src.actions.v16.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v16.implementation.peeling import Peeling

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v16")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per dim: [0, 4, 8, 16, 32].

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
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: packed_sizes (list[int]). Values per dim: [0, 4, 8, 16, 32].

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
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per dim: [0, 4, 8, 16, 32].

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
    Applies Vectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]). Values per dim: [0, 2, 4, 8, 16].

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
def loop_interchange_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Loop Interchange action on MLIR Code.

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
def loop_unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Loop Unrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_size (int from [4, 8, 16, 32]), unroll_factor (int from [2, 3, 4]).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = LoopUnrolling()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Parallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: num_threads (int from [2, 4, 8, 16, 28]).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Parallelization()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def im2col_lowering_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Im2col Lowering action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: enable (int, 0 or 1).

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
def peeling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Peeling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_size (int from [4, 8, 16, 32]), peel_front (int, 0 or 1).

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
