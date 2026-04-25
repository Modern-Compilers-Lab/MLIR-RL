from llm_action.src.actions.v14.implementation.tiling import Tiling
from llm_action.src.actions.v14.implementation.loop_interchange import LoopInterchange
from llm_action.src.actions.v14.implementation.packing import Packing
from llm_action.src.actions.v14.implementation.promotion import Promotion
from llm_action.src.actions.v14.implementation.vectorization import Vectorization
from llm_action.src.actions.v14.implementation.loop_unrolling import LoopUnrolling
from llm_action.src.actions.v14.implementation.peeling import Peeling
from llm_action.src.actions.v14.implementation.padding import Padding
from llm_action.src.actions.v14.implementation.parallelization import Parallelization
from llm_action.src.actions.v14.implementation.fusion import Fusion
from llm_action.src.actions.v14.implementation.loop_distribution import LoopDistribution
from llm_action.src.actions.v14.implementation.canonicalization import Canonicalization
from llm_action.src.actions.v14.implementation.im2col_lowering import Im2colLowering
from llm_action.src.actions.v14.implementation.bufferization_strategy import BufferizationStrategy

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space-v14")


@mcp.tool()
def tiling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Tiling action on MLIR Code.

    Partitions the iteration space of the target linalg op (tag = "operation_0")
    into smaller blocks via transform.structured.tile_using_for.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: tile_sizes (list[int]). Values per slot: [0, 16, 32, 64, 128].
            A value of 0 means do not tile this loop.

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
    Applies Loop Interchange action on MLIR Code.

    Generalizes the target linalg op and permutes its iteration-space loops via
    transform.structured.interchange with iterator_interchange.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: permutation (list[int]). Must be a permutation of range(n_loops),
            different from the identity.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopInterchange.precondition(code, parameters):
        transformed_code = LoopInterchange.implement(code, parameters)
        return True, transformed_code, LoopInterchange.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def packing_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Packing action on MLIR Code.

    Packs target operands into a blocked layout using transform.structured.pack,
    followed by lower_pack and lower_unpack.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: packed_sizes (list[int]). Values per slot: [0, 8, 16, 32, 64].
            A value of 0 means this dimension is not packed.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Packing.precondition(code, parameters):
        transformed_code = Packing.implement(code, parameters)
        return True, transformed_code, Packing.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def promotion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Promotion action on MLIR Code.

    Tiles the target linalg op and promotes the destination operand of the
    inner sub-tile into a local allocation via
    transform.structured.bufferize_to_allocation {bufferize_destination_only}.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: tile_sizes (list[int]). Values per slot: [0, 8, 16, 32, 64].
            A value of 0 means do not tile this loop.

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
    Applies Vectorization action on MLIR Code.

    Generalizes and tiles the target linalg op with static vector widths, then
    runs transform.structured.vectorize_children_and_apply_patterns on the
    enclosing function. The product of vector sizes is bounded by
    VECTORIZATION_SIZE_LIMIT.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: vector_sizes (list[int]). Values per slot: [1, 2, 4, 8, 16].
            Must not be all ones.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Vectorization.precondition(code, parameters):
        transformed_code = Vectorization.implement(code, parameters)
        return True, transformed_code, Vectorization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_unrolling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Loop Unrolling action on MLIR Code.

    Tiles the target linalg op and unrolls the innermost resulting scf.for
    loop via transform.loop.unroll with a fixed factor.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys:
              - tile_sizes (list[int]): Values [4, 8, 16, 32, 64].
              - unroll_factor (int): Values [2, 4, 8, 16, 32].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopUnrolling.precondition(code, parameters):
        transformed_code = LoopUnrolling.implement(code, parameters)
        return True, transformed_code, LoopUnrolling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def peeling_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Peeling action on MLIR Code.

    Tiles the target linalg op and peels the deepest non-trivial inner scf.for
    loop via transform.loop.peel, producing a main loop and a remainder loop.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: tile_sizes (list[int]). Values per slot: [8, 16, 24, 48, 96].
            At least one entry must be > 1.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Peeling.precondition(code, parameters):
        transformed_code = Peeling.implement(code, parameters)
        return True, transformed_code, Peeling.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def padding_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Padding action on MLIR Code.

    Pads the iteration space of the target linalg op to a multiple of a
    hardware-friendly constant via transform.structured.pad with
    copy_back_op = "none".

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: pad_to_multiple_of (list[int]).
            Values per slot: [1, 16, 32, 48, 64]. A value of 1 means no
            padding on that dimension.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Padding.precondition(code, parameters):
        transformed_code = Padding.implement(code, parameters)
        return True, transformed_code, Padding.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def parallelization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Parallelization action on MLIR Code.

    Tiles the target linalg op into an scf.forall loop via
    transform.structured.tile_using_forall, enabling coarse-grain parallel
    execution over independent tiles.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: tile_sizes (list[int]). Values per slot: [0, 8, 16, 32, 64].
            A value of 0 means do not parallelize this loop.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Parallelization.precondition(code, parameters):
        transformed_code = Parallelization.implement(code, parameters)
        return True, transformed_code, Parallelization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def fusion_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Fusion action on MLIR Code.

    Tiles the target linalg op and greedily fuses producer ops into the
    resulting tile loop nest via transform.structured.fuse.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: tile_sizes (list[int]). Values per slot: [0, 8, 16, 32, 64].
            A value of 0 means do not tile/fuse this loop.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Fusion.precondition(code, parameters):
        transformed_code = Fusion.implement(code, parameters)
        return True, transformed_code, Fusion.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def loop_distribution_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Loop Distribution action on MLIR Code.

    Distributes the target linalg op into two independent halves by splitting
    along a chosen dimension via transform.structured.split followed by
    transform.split_handle.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys:
              - dimension (int): Values [0, 1, 2, 3, 4].
              - chunk_size (int): Values [8, 16, 32, 64, 128].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if LoopDistribution.precondition(code, parameters):
        transformed_code = LoopDistribution.implement(code, parameters)
        return True, transformed_code, LoopDistribution.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def canonicalization_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Canonicalization action on MLIR Code.

    Runs transform.apply_patterns.canonicalization on the enclosing function,
    optionally interleaved with CSE.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: apply_cse (int). Values: [0, 1]. If 1, interleave CSE.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Canonicalization.precondition(code, parameters):
        transformed_code = Canonicalization.implement(code, parameters)
        return True, transformed_code, Canonicalization.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def im2col_lowering_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Im2col Lowering action on MLIR Code.

    Converts a linalg.conv_2d_* target op into an explicit im2col contraction
    via transform.structured.convert_conv2d_to_img2col. Only applicable when
    the payload contains a linalg.conv_2d op.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: enable (int). Values: [0, 1]. 1 runs the conversion.

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if Im2colLowering.precondition(code, parameters):
        transformed_code = Im2colLowering.implement(code, parameters)
        return True, transformed_code, Im2colLowering.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


@mcp.tool()
def bufferization_strategy_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies Bufferization Strategy action on MLIR Code.

    Chooses how tensor-semantic operands of the target linalg op are converted
    into buffer-semantic operands:
      - 0 ("eliminate"): eliminate_empty_tensors + empty_tensor_to_alloc_tensor.
      - 1 ("alloc-destination"): bufferize_to_allocation {bufferize_destination_only}.

    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action.
            Keys: strategy (int). Values: [0, 1].

    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    if BufferizationStrategy.precondition(code, parameters):
        transformed_code = BufferizationStrategy.implement(code, parameters)
        return True, transformed_code, BufferizationStrategy.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False


if __name__ == "__main__":
    mcp.run()
