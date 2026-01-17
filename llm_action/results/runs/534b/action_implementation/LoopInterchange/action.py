from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re

class LoopInterchange(ActionBase):
    """
    Loop Interchange Action:
    Permutes the iteration order of loops in a structured operation (linalg.matmul,
    linalg.conv_2d_*, linalg.generic, etc.) to improve memory access patterns.
    
    The transformation works by:
    1. Tiling the operation to expose explicit scf.for loops.
    2. Applying loop permutation via the `interchange` parameter of `tile_using_for`.
    
    The target operation is identified by the tag `operation_0`.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "loop_indices": {
                "type": "list[int]",
                "description": "Permutation of loop indices (e.g., [1, 0, 2] swaps first two loops)",
                "default": None,
            },
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes for each loop. Defaults to all 1s if not provided.",
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if the action is applicable:
        - Operation tagged with 'operation_0' exists.
        - loop_indices is a valid permutation.
        - loop_indices is not the identity permutation.
        """
        # Check for tagged operation
        if 'tag = "operation_0"' not in code:
            return False

        loop_indices = params.get("loop_indices")
        if loop_indices is None or not isinstance(loop_indices, list):
            return False

        # Validate permutation: all indices unique, in range [0, len-1)
        n = len(loop_indices)
        if n == 0:
            return False
        if sorted(loop_indices) != list(range(n)):
            return False

        # Reject identity permutation (no-op)
        if loop_indices == list(range(n)):
            return False

        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        No preprocessing needed; the transform dialect handles everything.
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Apply loop interchange via structured.tile_using_for with interchange parameter.
        
        Strategy:
        1. Infer the rank of the operation from the loop_indices.
        2. If tile_sizes not provided, default to all 1s.
        3. Construct a transform dialect snippet that:
           - Matches operation_0.
           - Tiles it with the given sizes and permutation.
        4. Execute the transform.
        """
        loop_indices = params.get("loop_indices")
        tile_sizes = params.get("tile_sizes")
        
        rank = len(loop_indices)
        
        # Default tile sizes to all 1s if not provided
        if tile_sizes is None:
            tile_sizes = [1] * rank
        
        # Ensure tile_sizes has correct length
        if len(tile_sizes) != rank:
            tile_sizes = tile_sizes[:rank] + [1] * max(0, rank - len(tile_sizes))
        
        # Construct the transform code
        # Format the tile_sizes and interchange arrays for MLIR
        tile_sizes_str = "[" + ", ".join(map(str, tile_sizes)) + "]"
        loop_indices_str = "[" + ", ".join(map(str, loop_indices)) + "]"
        num_loops = sum(1 for s in tile_sizes if s != 0)
        
        # Generate loop results syntax
        if num_loops == 0:
            num_loops = rank
        loop_results = ", ".join(["!transform.any_op"] * num_loops)
        
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f"    %op = transform.structured.match attributes{{tag = \"operation_0\"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled, %loops:{num_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes_str} interchange = {loop_indices_str} : (!transform.any_op) -> (!transform.any_op, {loop_results})\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )
        
        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception as e:
            # If transform fails, return original code; postcondition will catch it
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that:
        1. The IR changed (not a no-op).
        2. The IR is still valid and executable (contains func.func @main).
        """
        # Reject no-op transformations
        if before.strip() == after.strip():
            return False

        # Check that main function still exists
        if "func.func @main" not in after:
            return False

        # Check that IR is not empty or malformed
        if not after.strip() or "module" not in after:
            return False

        return True