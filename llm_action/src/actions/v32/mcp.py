from llm_action.src.actions.v32.implementation.tiling import Tiling
from llm_action.src.actions.v32.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v32.implementation.promotion import Promotion
from llm_action.src.actions.v32.implementation.vectorization import Vectorization
from llm_action.src.actions.v32.implementation.packing import Packing
from llm_action.src.actions.v32.implementation.parallelization import Parallelization
from llm_action.src.actions.v32.implementation.parallel_tiling import ParallelTiling

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v32")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            tile_sizes (list[int]): Per-loop tile sizes. 0 = do not tile that dimension.
            Values: [0, 16, 32, 64, 128].

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
    Applies LoopInterchange action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            permutation (list[int]): Loop dimension permutation. Must be a valid permutation
            of [0..n_loops-1], not the identity.

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
    Applies Promotion action on MLIR code.

    Args:
        code (str): The MLIR code (must be tensor-level, not yet bufferized).
        parameters (dict): Parameters for the action.
            tile_sizes (list[int]): Per-loop tile sizes for the pre-promotion tiling.
            0 = do not tile that dimension. Values: [0, 16, 32, 64, 128].

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
    Applies Vectorization action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            tile_sizes (list[int]): Per-loop tile/vector sizes. All must be > 0.
            Product must be <= VECTORIZATION_SIZE_LIMIT. Values: [4, 8, 16, 32].

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
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            packed_sizes (list[int]): Per-loop-dimension pack sizes. 0 = do not pack.
            Values: [0, 16, 32, 64, 128].

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
def parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Parallelization action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            num_threads (list[int]): Thread counts per parallel dimension [M, N].
            Valid configs: [28,1], [14,2], [7,4], [4,7], [2,14].

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
def parallel_tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelTiling action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            tile_sizes (list[int]): Per-loop tile sizes for forall-based parallel dispatch.
            0 = do not tile that dimension. Values: [0, 32, 64, 128, 256].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelTiling()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
