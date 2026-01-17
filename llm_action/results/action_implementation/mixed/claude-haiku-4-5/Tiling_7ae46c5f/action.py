from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Action(ActionBase):
    """
    Tiling Action: Partitions loop nests into smaller blocks to fit in L1/L2 cache.
    
    For structured operations (linalg.matmul, linalg.conv_2d, etc.), this action
    applies transform.structured.tile_using_for with the provided tile sizes.
    
    Preconditions:
    - The operation must be tagged with tag = "operation_0"
    - tile_sizes must be a non-empty list of non-negative integers
    - At least one tile size must be non-zero (no all-zero no-op)
    
    Postconditions:
    - The IR must be modified (not a no-op)
    - The IR must remain valid and executable
    - Tag may be preserved on the tiled operation
    """
    
    @classmethod
    def parameters(cls) -> dict:
        """Define action parameters: tile_sizes list."""
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes for each loop dimension. 0 means no tiling for that loop.",
                "default": None,  # Must be provided; no sensible default for general case
            }
        }
    
    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if tiling can be applied to this code.
        
        Returns False if:
        - The operation tag "operation_0" is missing
        - tile_sizes is missing, not a list, or empty
        - tile_sizes contains negative values
        - All tile sizes are zero (no-op)
        """
        # Check tag existence
        if 'tag = "operation_0"' not in code:
            return False
        
        # Check parameters
        if "tile_sizes" not in params:
            return False
        
        tile_sizes = params["tile_sizes"]
        
        # Must be a list
        if not isinstance(tile_sizes, list):
            return False
        
        # Must be non-empty
        if len(tile_sizes) == 0:
            return False
        
        # All elements must be non-negative integers
        try:
            for size in tile_sizes:
                if not isinstance(size, int) or size < 0:
                    return False
        except (TypeError, ValueError):
            return False
        
        # At least one must be non-zero (avoid no-op)
        if all(s == 0 for s in tile_sizes):
            return False
        
        return True
    
    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocessing: identity transformation.
        No canonicalization needed for tiling.
        """
        return code
    
    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Apply tiling via MLIR Transform dialect.
        
        Constructs a transform.structured.tile_using_for call with the provided
        tile_sizes. The transform matches the operation via tag = "operation_0"
        and applies tiling to each of the specified dimensions.
        """
        tile_sizes = params["tile_sizes"]
        
        # Generate the loop count string for the return types
        # tile_using_for returns: (tiled_op, loop0, loop1, ..., loopN)
        num_loops = len(tile_sizes)
        loop_returns = ", ".join(["!transform.any_op"] * num_loops)
        
        # Construct the transform dialect code
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op, %loops:{num_loops} = transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_returns})\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )
        
        # Apply the transform
        try:
            transformed = run_transform_code(code, transform_code)
            return transformed
        except Exception:
            # If transform fails, return original code
            # Postcondition will detect this as a no-op
            return code
    
    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that tiling was successfully applied.
        
        Returns False if:
        - The IR did not change (no-op)
        - The result is empty or malformed
        - func.func is missing (indicates IR corruption)
        """
        # Check that IR changed
        if before.strip() == after.strip():
            return False
        
        # Check that result is non-empty
        if not after or not after.strip():
            return False
        
        # Verify basic IR structure is preserved
        if "func.func" not in after:
            return False
        
        # Verify module structure exists
        if "module" not in after:
            return False
        
        return True