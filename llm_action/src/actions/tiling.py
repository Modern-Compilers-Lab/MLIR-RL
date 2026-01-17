# From llm_action/results/action_implementation/mixed/claude-haiku-4-5/Tiling_a2695da3/action.py

from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Tiling(ActionBase):
    """
    Tiling Action: Partitions loop nests into smaller rectangular blocks to fit
    intermediate results in L1/L2 cache for improved cache locality and memory efficiency.
    
    This action applies transform.structured.tile_using_for to the operation tagged
    with tag="operation_0", using the provided tile_sizes parameter to control the
    granularity of tiling along each loop dimension.
    
    Tile sizes of 0 mean no tiling for that dimension (the loop is untiled).
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Define the parameters for the Tiling action.
        
        Returns:
            dict: Parameter specification with 'tile_sizes' as the primary tunable.
        """
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "List of tile sizes for each loop dimension. Zero means no tiling.",
                "default": [8, 8, 8],
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if the action can be applied to the given IR.
        
        Preconditions:
        1. The code must contain tag="operation_0" in a structured linalg operation.
        2. tile_sizes must be a non-empty list of non-negative integers.
        3. At least one tile size must be non-zero (to avoid no-op).
        
        Args:
            code (str): The MLIR code to check.
            params (dict): Parameters including 'tile_sizes'.
        
        Returns:
            bool: True if the action is applicable, False otherwise.
        """
        # Check that tag="operation_0" exists
        if 'tag = "operation_0"' not in code:
            return False
        
        # Validate tile_sizes parameter
        tile_sizes = params.get("tile_sizes", [])
        
        # Must be a list
        if not isinstance(tile_sizes, list):
            return False
        
        # Must be non-empty
        if len(tile_sizes) == 0:
            return False
        
        # All elements must be non-negative integers
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
        
        # At least one tile size must be non-zero (to avoid trivial no-op)
        if all(s == 0 for s in tile_sizes):
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocess the code before transformation.
        
        For tiling, no preprocessing is required. The Transform dialect
        handles loop structure discovery and tiling automatically.
        
        Args:
            code (str): The MLIR code.
            params (dict): Parameters (unused).
        
        Returns:
            str: The unchanged code.
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Implement the tiling transformation using MLIR Transform dialect.
        
        Constructs and executes a Transform dialect sequence that:
        1. Matches the operation tagged with tag="operation_0".
        2. Applies transform.structured.tile_using_for with the provided tile_sizes.
        
        Args:
            code (str): The MLIR code to transform.
            params (dict): Parameters including 'tile_sizes'.
        
        Returns:
            str: The tiled MLIR code, or the original code if tiling fails.
        """
        tile_sizes = params.get("tile_sizes", [])
        
        # Count non-zero tile sizes to determine number of loop variables in result
        num_tiles = sum(1 for s in tile_sizes if s != 0)
        
        # Build the return type for transform.structured.tile_using_for
        # Format: %tiled_op, %tile0, %tile1, ... = tile_using_for ...
        if num_tiles > 0:
            tile_vars = ", ".join([f"%tile{i}" for i in range(num_tiles)])
            result_types = ", ".join(["!transform.any_op"] * (1 + num_tiles))
            tile_result = f"%tiled_op, {tile_vars} = "
        else:
            # If all zeros (shouldn't happen due to precondition), just tile
            result_types = "!transform.any_op"
            tile_result = "%tiled_op = "
        
        # Construct the Transform dialect code
        transform_code = (
            "module attributes {transform.with_named_sequence} {\n"
            "  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {\n"
            '    %op = transform.structured.match attributes{tag = "operation_0"} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f"    {tile_result}transform.structured.tile_using_for %op tile_sizes {tile_sizes} : (!transform.any_op) -> ({result_types})\n"
            "    transform.yield\n"
            "  }\n"
            "}\n"
        )
        
        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception:
            # If transformation fails, return original code
            # (postcondition will detect no-op and fail appropriately)
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the transformation succeeded and was not a no-op.
        
        Postconditions:
        1. The after code must not be identical to before (reject no-ops).
        2. The after code must still be valid MLIR (contain func.func).
        
        Args:
            before (str): The original MLIR code.
            after (str): The transformed MLIR code.
            params (dict): Parameters (unused).
        
        Returns:
            bool: True if transformation succeeded, False if no-op or invalid.
        """
        # Reject if no changes were made (no-op)
        if before.strip() == after.strip():
            return False
        
        # Verify the result is still valid MLIR (basic sanity check)
        if "func.func" not in after:
            return False
        
        return True