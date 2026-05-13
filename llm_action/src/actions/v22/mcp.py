from llm_action.src.actions.v22.implementation.tiling import Tiling
from llm_action.src.actions.v22.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v22.implementation.promotion import Promotion
from llm_action.src.actions.v22.implementation.vectorization import Vectorization
from llm_action.src.actions.v22.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v22.implementation.packing import Packing
from llm_action.src.actions.v22.implementation.parallelization import Parallelization
from llm_action.src.actions.v22.implementation.loop_peeling import LoopPeeling

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v22")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Required key: "tile_sizes" (list[int]) — tile size per loop dimension; 0 means do not tile. Values: [0, 4, 8, 16, 32].

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
def loop_interchange_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Loop Interchange action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Required key: "permutation" (list[int]) — permutation of loop iterator indices. E.g. [1, 2, 0].

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
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Required key: "operands_to_promote" (list[int]) — operand indices to promote. Values: [0,1,2], [0,1], [0,2], [1,2], [1].

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
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Required key: "vector_sizes" (list[int]) — SIMD vector width per dimension. Values: [2, 4, 8, 16].

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
def loop_unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Loop Unrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Required key: "unroll_factor" (int) — number of loop body copies per iteration. Values: [2, 4, 8].

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
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Required key: "packed_sizes" (list[int]) — inner block dimensions for packing. Values: [0, 8, 16, 32, 64].

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
def parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Parallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Required key: "num_threads" (int) — number of threads. Values: [2, 4, 8, 16, 32].

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
def loop_peeling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Loop Peeling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. No tunable parameters — pass empty dict {}.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = LoopPeeling
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
