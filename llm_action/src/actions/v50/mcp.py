from llm_action.src.actions.v50.implementation.tiling import Tiling
from llm_action.src.actions.v50.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v50.implementation.promotion import Promotion
from llm_action.src.actions.v50.implementation.sequential_vectorization import SequentialVectorization
from llm_action.src.actions.v50.implementation.parallel_vectorization import ParallelVectorization
from llm_action.src.actions.v50.implementation.tiling_parallelization import TilingParallelization
from llm_action.src.actions.v50.implementation.thread_parallelization import ThreadParallelization

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) — per-loop tile sizes; 0 means do not tile. Values from [0, 4, 8, 16, 32, 64].

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
        parameters (dict): Parameters for the action. Keys: "permutation" (list[int]) — permutation of loop indices. Must be a non-identity permutation of range(n_loops).

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
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) — per-loop tile sizes for prerequisite tiling; 0 = skip. Values from [0, 4, 8, 16, 32, 64]. "operands_to_promote" (list[int]) — operand indices to promote (e.g. [0, 1, 2]).

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
def sequential_vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies SequentialVectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "vector_sizes" (list[int]) — per-dimension tile sizes for SIMD-width tiling; all dims are tiled. Values from [1, 4, 8, 16, 32, 64].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = SequentialVectorization()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallel_vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ParallelVectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "vector_sizes" (list[int]) — per-dimension tile sizes for forall distribution; 0 = skip. Values from [0, 4, 8, 16, 32, 64].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ParallelVectorization()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def tiling_parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies TilingParallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) — per-parallel-dimension tile sizes for forall distribution; 0 = skip. Values from [0, 4, 8, 16, 32, 64].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = TilingParallelization()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def thread_parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ThreadParallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "num_threads" (int) — number of threads. Values from [2, 4, 8, 16, 32, 64].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = ThreadParallelization()
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
