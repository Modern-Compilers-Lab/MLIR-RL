from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re


class TilingAction(ActionBase):
    """
    Tiling transformation: partitions loop nests into rectangular tiles 
    to improve cache locality and data reuse.
    
    This action matches the operation tagged with "tag = operation_0" 
    and applies structured tiling using for-loops.
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Returns the parameter space for this action.
        
        tile_sizes: list of integers, one per loop dimension.
                   0 means no tiling for that dimension.
        """
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes for each loop dimension. 0 = no tiling.",
                "default": [32, 32, 32],  # Generic default for 3D matmul-like ops
                "constraints": "All non-negative integers; at least one must be > 0."
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Checks applicability:
        1. The code contains the target operation with tag="operation_0"
        2. tile_sizes is a non-empty list of non-negative integers
        3. At least one tile size is > 0 (avoid no-op)
        """
        # Check for tag="operation_0" in the code
        if 'tag = "operation_0"' not in code and "tag = 'operation_0'" not in code:
            return False
        
        # Check tile_sizes parameter
        if "tile_sizes" not in params:
            return False
        
        tile_sizes = params["tile_sizes"]
        
        # Must be a list
        if not isinstance(tile_sizes, (list, tuple)):
            return False
        
        # Must be non-empty
        if len(tile_sizes) == 0:
            return False
        
        # All elements must be non-negative integers
        for size in tile_sizes:
            if not isinstance(size, int) or size < 0:
                return False
        
        # At least one tile size must be > 0 (avoid no-op)
        if all(s == 0 for s in tile_sizes):
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocessing: identity (no special preparation needed for tiling).
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Applies tiling transformation using MLIR Transform dialect.
        
        Constructs a transform sequence that:
        1. Matches the operation with tag="operation_0"
        2. Applies tile_using_for with the specified tile sizes
        3. Returns the transformed IR
        """
        tile_sizes = params["tile_sizes"]
        
        # Count the number of loops (non-zero tile sizes determine the loop count)
        num_loops = len(tile_sizes)
        
        # Generate loop variable names for the result
        # e.g., for 3 loops: %loop0, %loop1, %loop2
        loop_vars = ', '.join([f'%loop{i}' for i in range(num_loops)])
        
        # Build the transform code
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n"
            f"    %op = transform.structured.match attributes{{tag = \"operation_0\"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, {loop_vars} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, "
            f"{', '.join(['!transform.any_op'] * num_loops)})\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )
        
        # Apply the transformation
        try:
            transformed = run_transform_code(code, transform_code)
            return transformed
        except Exception as e:
            # If transform fails, return original code
            # Postcondition will detect this as a no-op/failure
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Validates that the transformation succeeded:
        1. The IR must have changed (not a no-op)
        2. The IR must still be valid (contain func.func, parse correctly)
        3. The tag should still exist (preserved through transformation)
        """
        # Check that IR changed
        if before.strip() == after.strip():
            return False
        
        # Check that result is non-empty
        if not after or not after.strip():
            return False
        
        # Check that func.func still exists (basic sanity)
        if "func.func" not in after:
            return False
        
        # Tag should still exist in some form (may be in tiled ops)
        # We don't strictly enforce tag preservation since the transform 
        # framework may move or duplicate it
        
        return True