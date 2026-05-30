from llm_action.src.actions.v38.implementation.tiling import Tiling
from llm_action.src.actions.v38.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v38.implementation.promotion import Promotion
from llm_action.src.actions.v38.implementation.vectorization import Vectorization
from llm_action.src.actions.v38.implementation.parallel_vectorization import ParallelVectorization
from llm_action.src.actions.v38.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v38.implementation.tiling_based_parallelization import TilingBasedParallelization
from llm_action.src.actions.v38.implementation.direct_parallelization import DirectParallelization

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per slot: [0, 4, 8, 16, 32].

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
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]), operands_to_promote (list[int]). Tile values: [0, 16, 32, 64, 128]. Operand options: [0,1,2], [0,1], [0], [1], [2].

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
    Applies Vectorization action on MLIR Code (sequential tiling + vectorize).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]). Values per slot: [1, 2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Vectorization.precondition(code, parameters):
        transformed_code = Vectorization.implement(code, parameters)
        return True, transformed_code, Vectorization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallel_vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelVectorization action on MLIR Code (parallel tiling + vectorize).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]). Values per slot: [1, 2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if ParallelVectorization.precondition(code, parameters):
        transformed_code = ParallelVectorization.implement(code, parameters)
        return True, transformed_code, ParallelVectorization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopUnrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: unroll_factor (int). Values: [2, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopUnrolling.precondition(code, parameters):
        transformed_code = LoopUnrolling.implement(code, parameters)
        return True, transformed_code, LoopUnrolling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def tiling_based_parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies TilingBasedParallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per slot: [0, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if TilingBasedParallelization.precondition(code, parameters):
        transformed_code = TilingBasedParallelization.implement(code, parameters)
        return True, transformed_code, TilingBasedParallelization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def direct_parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies DirectParallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: num_threads (int). Values: [2, 4, 8, 16, 28].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if DirectParallelization.precondition(code, parameters):
        transformed_code = DirectParallelization.implement(code, parameters)
        return True, transformed_code, DirectParallelization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
