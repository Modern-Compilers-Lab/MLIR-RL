from llm_action.src.actions.v21.implementation.tiling import Tiling
from llm_action.src.actions.v21.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v21.implementation.promotion import Promotion
from llm_action.src.actions.v21.implementation.vectorization import Vectorization
from llm_action.src.actions.v21.implementation.unrolling import Unrolling
from llm_action.src.actions.v21.implementation.packing import Packing
from llm_action.src.actions.v21.implementation.parallelization import Parallelization
from llm_action.src.actions.v21.implementation.peeling import Peeling

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v21")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code. Partitions the iteration space into
    fixed-size blocks for cache locality.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) - tile sizes per loop dimension, 0 means do not tile. Values from [0, 4, 8, 16, 32].

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
    Applies Loop Interchange action on MLIR Code. Permutes loop order to
    improve memory access patterns.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "permutation" (list[int]) - non-identity permutation of loop indices [0..n-1].

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
    Applies Promotion action on MLIR Code. Copies tiled operands into contiguous
    local buffers to eliminate irregular strides.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_sizes" (list[int]) - tile sizes for outer tiling before promotion. Values from [4, 8, 16, 32, 64].

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
    Applies Vectorization action on MLIR Code. Maps loop iterations onto SIMD
    vector lanes for packed vector operations.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "vector_sizes" (list[int]) - vector sizes per dimension. Values from [2, 4, 8, 16, 32]. Product must be <= 1024.

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
def unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Unrolling action on MLIR Code. Replicates loop body to reduce
    overhead and increase instruction-level parallelism.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_size" (int) from [4, 8, 16, 32, 64], "unroll_factor" (int) from [2, 3, 4]. tile_size must be divisible by unroll_factor.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Unrolling
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code. Rearranges data layout for contiguous
    and aligned memory access by vectorized loops.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "packed_sizes" (list[int]) - packed sizes per iterator dimension. 0 means do not pack. Values from [0, 4, 8, 16, 32].

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
    Applies Parallelization action on MLIR Code. Distributes iterations across
    CPU threads using scf.forall.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "num_threads" (list[int]) - number of threads per parallel dimension. Values from [2, 4, 8, 16, 32].

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
def peeling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Peeling action on MLIR Code. Separates partial/remainder iterations
    into a separate epilogue loop so the main loop operates on full tiles.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action. Keys: "tile_size" (int) - tile size that may not evenly divide iteration space. Values from [24, 48, 36, 20, 40].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Peeling
    if action.precondition(code, parameters):
        transformed_code = action.implement(code, parameters)
        return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
