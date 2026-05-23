from llm_action.src.actions.v37.implementation.tiling import Tiling
from llm_action.src.actions.v37.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v37.implementation.vectorization import Vectorization
from llm_action.src.actions.v37.implementation.unrolling import Unrolling
from llm_action.src.actions.v37.implementation.parallelization import Parallelization
from llm_action.src.actions.v37.implementation.parallelization_by_num_threads import ParallelizationByNumThreads

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v37")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            tile_sizes (list[int]): Per-loop tile sizes. 0 = do not tile that dimension.
            Values: [0, 4, 8, 16, 32].

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
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            tile_sizes (list[int]): Per-loop tile/vector sizes. All must be > 0.
            Product must be <= 2048. Values: [1, 2, 4, 8, 16].

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
    Applies Unrolling action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            loop_dim (int): Which loop dimension to unroll (0-indexed).
            unroll_factor (int): Unroll factor. Values: [2, 4, 8, 16].

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
def parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Parallelization action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            tile_sizes (list[int]): Per-loop tile sizes for forall-based parallel dispatch.
            0 = do not tile that dimension. Values: [0, 32, 64, 128, 256].

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
def parallelization_by_num_threads_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationByNumThreads action on MLIR code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            num_threads (int): Number of threads to distribute work across.
            Values: [2, 4, 8, 14, 28].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelizationByNumThreads()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
