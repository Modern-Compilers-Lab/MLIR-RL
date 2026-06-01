from llm_action.src.actions.v53.implementation.tiling import Tiling
from llm_action.src.actions.v53.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v53.implementation.promotion import Promotion
from llm_action.src.actions.v53.implementation.sequential_vectorization import SequentialVectorization
from llm_action.src.actions.v53.implementation.parallel_vectorization import ParallelVectorization
from llm_action.src.actions.v53.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v53.implementation.tiling_parallelization import TilingParallelization
from llm_action.src.actions.v53.implementation.thread_parallelization import ThreadParallelization

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per slot: [0, 4, 8, 16, 32, 64].

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
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per slot: [0, 4, 8, 16, 32, 64].

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
    Applies Sequential Vectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]). Values per slot: [1, 2, 4, 8, 16, 32].

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
    Applies Parallel Vectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: vector_sizes (list[int]). Values per slot: [1, 2, 4, 8, 16, 32].

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
def im2col_lowering_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Im2col Lowering action on MLIR Code. Converts conv2d to matmul-like contraction.

    Args:
        code (str): The MLIR code containing a conv_2d_nchw_fchw operation.
        parameters (dict): No parameters needed (empty dict).

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
def tiling_parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling Parallelization action on MLIR Code (tile_using_forall).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: tile_sizes (list[int]). Values per slot: [0, 4, 8, 16, 32, 64].

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
    Applies Thread Parallelization action on MLIR Code (forall with num_threads).

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: num_threads (int). Values: [2, 4, 8, 16, 32].

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
