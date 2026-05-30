from llm_action.src.actions.v41.implementation.tiling import Tiling
from llm_action.src.actions.v41.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v41.implementation.promotion import Promotion
from llm_action.src.actions.v41.implementation.vectorization_seq import VectorizationSeq
from llm_action.src.actions.v41.implementation.vectorization_par import VectorizationPar
from llm_action.src.actions.v41.implementation.unrolling import Unrolling
from llm_action.src.actions.v41.implementation.parallelization_tile import ParallelizationTile
from llm_action.src.actions.v41.implementation.parallelization_threads import ParallelizationThreads
from llm_action.src.actions.v41.implementation.packing import Packing

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            - tile_sizes (list[int]): Per-loop tile sizes (0=skip). Values: [0, 4, 8, 16, 32].

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
        parameters (dict): Parameters for the action.
            - permutation (list[int]): Loop permutation order.

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
        parameters (dict): Parameters for the action.
            - operands_to_promote (list[int]): Operand indices. Values: [[0,1,2], [0,1], [0], [1], [2]].
            - tile_sizes (list[int]): Tile sizes for promotion. Values: [8, 16, 32, 64, 128].

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
def vectorization_seq_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies VectorizationSeq action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            - vector_sizes (list[int]): Per-loop vector sizes. Values: [1, 2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = VectorizationSeq()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_par_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies VectorizationPar action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            - vector_sizes (list[int]): Per-loop vector sizes. Values: [1, 2, 4, 8, 16].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = VectorizationPar()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Unrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            - tile_sizes (list[int]): Tile sizes (factor at target dim, 0 elsewhere).
            - unroll_factor (int): Unroll factor. Values: [2, 4, 8, 16, 32].

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
def parallelization_tile_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationTile action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            - tile_sizes (list[int]): Per-loop tile sizes for parallel distribution (0=skip). Values: [0, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelizationTile()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_threads_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationThreads action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            - num_threads (int): Number of parallel threads. Values: [2, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelizationThreads()
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
        parameters (dict): Parameters for the action.
            - packed_sizes (list[int]): Per-dimension inner tile sizes (0=skip). Values: [0, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Packing()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
