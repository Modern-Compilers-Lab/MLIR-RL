from llm_action.src.actions.v52.implementation.tiling import Tiling
from llm_action.src.actions.v52.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v52.implementation.vectorization_sequential import VectorizationSequential
from llm_action.src.actions.v52.implementation.vectorization_parallel import VectorizationParallel
from llm_action.src.actions.v52.implementation.parallelization_tiling import ParallelizationTiling
from llm_action.src.actions.v52.implementation.parallelization_threads import ParallelizationThreads

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code using tile_using_for.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) — tile size per loop dim, 0 = skip. Values from [0, 4, 8, 16, 32, 64].

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
        parameters (dict): Parameters for the action. Keys: "permutation" (list[int]) — permutation of [0..n_loops-1].

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
def vectorization_sequential_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies VectorizationSequential action on MLIR Code (tile_using_for + vectorize).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "vector_sizes" (list[int]) — SIMD width per loop dim. Values from [1, 2, 4, 8].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = VectorizationSequential()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_parallel_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies VectorizationParallel action on MLIR Code (tile_using_forall + vectorize).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "vector_sizes" (list[int]) — SIMD width per loop dim. Values from [1, 2, 4, 8].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = VectorizationParallel()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationTiling action on MLIR Code using tile_using_forall with tile_sizes.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) — tile size per loop dim for parallel distribution, 0 = skip. Values from [0, 4, 8, 16, 32, 64].

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
def parallelization_threads_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelizationThreads action on MLIR Code using tile_using_forall with num_threads.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "num_threads" (int) — number of threads. Values from [2, 4, 8, 16, 32, 64].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelizationThreads()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
