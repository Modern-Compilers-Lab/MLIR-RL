from llm_action.src.actions.v24.implementation.tiling import Tiling
from llm_action.src.actions.v24.implementation.promotion import Promotion
from llm_action.src.actions.v24.implementation.packing import Packing
from llm_action.src.actions.v24.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v24.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v24.implementation.loop_peeling import LoopPeeling
from llm_action.src.actions.v24.implementation.vectorization import Vectorization
from llm_action.src.actions.v24.implementation.parallelization import Parallelization
from llm_action.src.actions.v24.implementation.canonicalization import Canonicalization

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v24")


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
    action = Tiling
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code (tiles, bufferizes, promotes operands to contiguous buffers).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per dim: [0, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Promotion
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code (reorganizes data layout with inner block dims).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: packed_sizes (list[int]). Values per dim: [0, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Packing
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_interchange_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopInterchange action on MLIR Code (reorders loop dimensions).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: permutation (list[int]).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = LoopInterchange
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopUnrolling action on MLIR Code (tiles reduction dim then unrolls).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: unroll_factor (int). Values: [2, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = LoopUnrolling
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_peeling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopPeeling action on MLIR Code (tiles reduction dim then peels remainder).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_size (int). Values: [3, 5, 7, 11, 13].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = LoopPeeling
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR Code (tiles then vectorizes to SIMD ops).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]). Values per dim: [2, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Vectorization
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Parallelization action on MLIR Code (distributes work across threads via scf.forall).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: num_threads (int). Values: [2, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Parallelization
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def canonicalization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Canonicalization action on MLIR Code (simplifies and normalizes IR).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action (empty dict, no tunable parameters).

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Canonicalization
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
