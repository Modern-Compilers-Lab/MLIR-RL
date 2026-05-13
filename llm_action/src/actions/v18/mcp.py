from llm_action.src.actions.v18.implementation.tiling import Tiling
from llm_action.src.actions.v18.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v18.implementation.promotion import Promotion
from llm_action.src.actions.v18.implementation.vectorization import Vectorization
from llm_action.src.actions.v18.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v18.implementation.parallelization import Parallelization
from llm_action.src.actions.v18.implementation.packing import Packing

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space")

@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) - tile size per loop dimension, 0 means no tiling. Values from [0, 4, 8, 16, 32].

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
    Applies LoopInterchange action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "permutation" (list[int]) - permutation of loop indices specifying new iterator order.

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
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) - tile sizes for the tiling step before promotion, 0 means no tiling. Values from [0, 8, 16, 32, 64].

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
        parameters (dict): Parameters for the action. Keys: "vector_sizes" (list[int]) - vector size per loop dimension. Values from [2, 4, 8, 16, 32]. Product must be <= 1024.

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
    Applies LoopUnrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_size" (int) - tile size for the innermost loop from [4, 8, 16, 32, 64], "unroll_factor" (int) - replication count from [2, 4, 8].

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
def parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Parallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "num_threads" (int) - number of threads from [2, 4, 8, 16, 32].

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
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "packed_sizes" (list[int]) - block dimensions for packing per iterator dimension, 0 means no packing. Values from [0, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Packing
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False

if __name__ == "__main__":
    mcp.run()
