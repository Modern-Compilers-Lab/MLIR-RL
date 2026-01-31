from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Tiling(ActionBase):
    """
    Tiling transformation action.
    
    Partitions the iteration space of the target operation (tagged with "operation_0")
    into rectangular blocks (tiles) to improve cache locality and data reuse.
    Uses transform.structured.tile_using_for for sequential tiling on CPU.
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Define the parameters for the tiling action.
        
        Returns:
            dict: Parameter specification with tile_sizes.
        """
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes for each loop dimension. 0 means no tiling for that dimension.",
                "default": None,
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if the action is applicable to the given IR.
        
        Validates:
        - The target operation with tag "operation_0" exists
        - tile_sizes parameter is well-formed and non-empty
        - At least one tile size is non-zero (non-trivial tiling)
        
        Args:
            code (str): MLIR IR as string.
            params (dict): Action parameters, must contain "tile_sizes".
        
        Returns:
            bool: True if action is applicable, False otherwise.
        """
        # Check that tag "operation_0" exists
        if 'tag = "operation_0"' not in code:
            return False
        
        # Extract and validate tile_sizes parameter
        if "tile_sizes" not in params:
            return False
        
        tile_sizes = params.get("tile_sizes")
        
        # Tile sizes must be a non-empty list
        if not isinstance(tile_sizes, list) or len(tile_sizes) == 0:
            return False
        
        # All elements must be non-negative integers
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
        
        # At least one tile size must be non-zero (no trivial/no-op tiling)
        if all(s == 0 for s in tile_sizes):
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Optional preprocessing / canonicalization.
        
        For tiling, no preprocessing is required. Return the code as-is.
        
        Args:
            code (str): MLIR IR as string.
            params (dict): Action parameters.
        
        Returns:
            str: Preprocessed (or identity) MLIR code.
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Implement the tiling transformation using MLIR Transform dialect.
        
        Constructs a transform sequence that:
        1. Matches the target operation via tag = "operation_0"
        2. Applies tile_using_for with the provided tile sizes
        
        Args:
            code (str): MLIR IR as string.
            params (dict): Action parameters with tile_sizes.
        
        Returns:
            str: Transformed MLIR code, or original code on failure.
        """
        tile_sizes = params["tile_sizes"]
        
        # Determine number of loops to be generated (count non-zero tile sizes)
        num_loops = sum(1 for s in tile_sizes if s != 0)
        
        # Build the result type for transform.structured.tile_using_for
        # Result is: tiled_op (1) + loops (num_loops)
        loop_types = ", ".join(["!transform.any_op"] * num_loops)
        result_type = f"(!transform.any_op, {loop_types})" if num_loops > 0 else "(!transform.any_op)"
        
        # Format tile sizes as MLIR dense array syntax
        tile_sizes_str = ", ".join(str(s) for s in tile_sizes)
        
        # Construct the transform IR
        transform_code = f"""module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg1
      : (!transform.any_op) -> !transform.any_op
    %tiled_op, %loops:{num_loops} = transform.structured.tile_using_for %op
      tile_sizes [{tile_sizes_str}]
      : (!transform.any_op) -> {result_type}
    transform.yield
  }}
}}"""
        
        # Apply the transform
        try:
            transformed_code = run_transform_code(code, transform_code)
            return transformed_code
        except Exception:
            # On any exception, return the original code
            # (postcondition will detect no-op and return False)
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the transformation succeeded.
        
        Checks:
        - The IR has changed (not a no-op)
        - The IR is still valid (contains func.func)
        - The IR is non-empty
        
        Args:
            before (str): Original MLIR code.
            after (str): Transformed MLIR code.
            params (dict): Action parameters.
        
        Returns:
            bool: True if transformation was successful, False otherwise.
        """
        # Reject no-op transformations
        if before.strip() == after.strip():
            return False
        
        # Check that the result is non-empty
        if not after or not after.strip():
            return False
        
        # Basic sanity check: the result should still be valid MLIR
        # (contain func.func, no obvious syntax errors)
        if "func.func" not in after:
            return False
        
        return True