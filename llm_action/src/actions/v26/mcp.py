from llm_action.src.actions.v26.implementation.tiling import Tiling
from llm_action.src.actions.v26.implementation.promotion import Promotion
from llm_action.src.actions.v26.implementation.packing import Packing
from llm_action.src.actions.v26.implementation.vectorization import Vectorization
from llm_action.src.actions.v26.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v26.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v26.implementation.tiling_based_parallelization import TilingBasedParallelization
from llm_action.src.actions.v26.implementation.thread_count_parallelization import ThreadCountParallelization
from llm_action.src.actions.v26.implementation.split_reduction import SplitReduction
from llm_action.src.actions.v26.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v26.implementation.loop_peeling import LoopPeeling
from llm_action.src.actions.v26.implementation.canonicalization import Canonicalization

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v26")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"tile_sizes": [int, int, int]} where each int is from [0, 4, 8, 16, 32]. Zero means no tiling on that dimension.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Tiling.precondition(code, parameters):
        transformed_code = Tiling.implement(code, parameters)
        return True, transformed_code, Tiling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"operands_to_promote": list[int]} from [[0,1,2], [0,1], [0], [1], [2]].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Promotion.precondition(code, parameters):
        transformed_code = Promotion.implement(code, parameters)
        return True, transformed_code, Promotion.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"packed_sizes": [int, int, int]} where each int is from [0, 2, 4, 8, 16]. Zero means no packing on that dimension.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Packing.precondition(code, parameters):
        transformed_code = Packing.implement(code, parameters)
        return True, transformed_code, Packing.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def vectorization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Vectorization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"vector_sizes": [int, int, int]} where each int is from [2, 4, 8, 16, 32]. Tiles then vectorizes. Product of sizes must be <= 1024.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Vectorization.precondition(code, parameters):
        transformed_code = Vectorization.implement(code, parameters)
        return True, transformed_code, Vectorization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_interchange_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopInterchange action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"permutation": list[int]} - a permutation of dimension indices.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopInterchange.precondition(code, parameters):
        transformed_code = LoopInterchange.implement(code, parameters)
        return True, transformed_code, LoopInterchange.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopUnrolling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"unroll_factor": int} from [2, 4, 8, 16, 32].

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
        parameters (dict): Parameters: {"tile_sizes": [int, int, int]} where each int is from [0, 4, 8, 16, 32]. Zero means no parallelization on that dimension.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if TilingBasedParallelization.precondition(code, parameters):
        transformed_code = TilingBasedParallelization.implement(code, parameters)
        return True, transformed_code, TilingBasedParallelization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def thread_count_parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies ThreadCountParallelization action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"num_threads": int} from [2, 4, 8, 14, 28].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if ThreadCountParallelization.precondition(code, parameters):
        transformed_code = ThreadCountParallelization.implement(code, parameters)
        return True, transformed_code, ThreadCountParallelization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def split_reduction_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies SplitReduction action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"split_factor": int} from [2, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if SplitReduction.precondition(code, parameters):
        transformed_code = SplitReduction.implement(code, parameters)
        return True, transformed_code, SplitReduction.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def im2col_lowering_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Im2colLowering action on MLIR Code. Converts conv2d to img2col + matmul.

    Args:
        code (str): The MLIR code containing linalg.conv_2d_nchw_fchw.
        parameters (dict): No parameters needed, pass {}.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Im2colLowering.precondition(code, parameters):
        transformed_code = Im2colLowering.implement(code, parameters)
        return True, transformed_code, Im2colLowering.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_peeling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies LoopPeeling action on MLIR Code.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters: {"tile_size": int} from [3, 6, 12, 24, 48].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopPeeling.precondition(code, parameters):
        transformed_code = LoopPeeling.implement(code, parameters)
        return True, transformed_code, LoopPeeling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def canonicalization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Canonicalization action on MLIR Code. Runs canonicalize + CSE passes.

    Args:
        code (str): The MLIR code.
        parameters (dict): No parameters needed, pass {}.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Canonicalization.precondition(code, parameters):
        transformed_code = Canonicalization.implement(code, parameters)
        return True, transformed_code, Canonicalization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
