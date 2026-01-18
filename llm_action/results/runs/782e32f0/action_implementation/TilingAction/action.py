from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class TilingAction(ActionBase):
    """
    Tiling transformation: partitions loop nest into cache-friendly tiles.
    
    Uses MLIR Transform dialect `transform.structured.tile_using_for` to 
    introduce nested loop levels that improve data locality and cache utilization.
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Returns the parameterization schema for the Tiling action.
        
        Returns:
            dict: A single parameter 'tile_sizes' (list of integers).
        """
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes per dimension. 0 = no tiling for that dimension.",
                "required": True,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check whether tiling is applicable to the given IR.
        
        Validates:
        - The target operation has tag = "operation_0"
        - tile_sizes is a list of non-negative integers
        - At least one tile size is non-zero (not a no-op)
        
        Args:
            code: The MLIR code (as string)
            params: Dictionary containing 'tile_sizes'
            
        Returns:
            bool: True if transformation is applicable, False otherwise.
        """
        # Check for presence of tag = "operation_0"
        if 'tag = "operation_0"' not in code:
            return False
        
        # Extract and validate tile_sizes parameter
        tile_sizes = params.get("tile_sizes")
        if not isinstance(tile_sizes, list):
            return False
        
        # All elements must be non-negative integers
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
        
        # At least one tile size must be non-zero (otherwise it's a no-op)
        if all(s == 0 for s in tile_sizes):
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Optional preprocessing. For tiling, no preprocessing is needed.
        
        Args:
            code: The MLIR code
            params: Dictionary containing 'tile_sizes'
            
        Returns:
            str: The code unchanged (identity preprocessing).
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Implements the tiling transformation using Transform dialect.
        
        Constructs a Transform IR snippet using `transform.structured.tile_using_for`
        to tile the operation tagged with "operation_0".
        
        The number of generated loop handles is equal to the count of non-zero
        tile sizes.
        
        Args:
            code: The MLIR payload code
            params: Dictionary containing 'tile_sizes' (list of integers)
            
        Returns:
            str: The transformed MLIR code (with tiled loop structure).
        """
        tile_sizes = params.get("tile_sizes")
        
        # Count non-zero tile sizes to determine number of loop results
        num_loops = sum(1 for s in tile_sizes if s != 0)
        
        # Build the loop result handles string
        # Format: %loops:N means N loop handles
        if num_loops == 0:
            # All zeros (should have been caught by precondition, but be safe)
            loop_spec = ""
        elif num_loops == 1:
            loop_spec = ", !transform.any_op"
        else:
            loop_spec = ", " + ", ".join(["!transform.any_op"] * num_loops)
        
        # Construct the Transform dialect code
        transform_ir = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f"    %op_0 = transform.structured.match attributes{{tag = \"operation_0\"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n"
            f"    %tiled_op, %loops:{num_loops} = transform.structured.tile_using_for %op_0 tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op{loop_spec})\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )
        
        # Apply the transform using the runtime helper
        try:
            transformed = run_transform_code(code, transform_ir)
            return transformed
        except Exception:
            # If transformation fails, return original code
            # Postcondition will catch this as a no-op
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Validates that the tiling transformation succeeded.
        
        Checks:
        - The code has changed (not a no-op)
        - The transformed code is non-empty and contains func.func
        
        Args:
            before: Original MLIR code
            after: Transformed MLIR code
            params: Dictionary containing 'tile_sizes'
            
        Returns:
            bool: True if transformation succeeded, False otherwise.
        """
        # Reject no-op transformations (code unchanged)
        if before.strip() == after.strip():
            return False
        
        # Basic sanity checks
        if not after or not after.strip():
            return False
        
        # Ensure the transformed code still has a module and func
        if "module" not in after or "func.func" not in after:
            return False
        
        return True