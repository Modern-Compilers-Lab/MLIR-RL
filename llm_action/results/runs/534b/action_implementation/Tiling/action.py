from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re


class Tiling(ActionBase):
    """
    Tiling action: Decompose loop nests into nested tile loops.
    
    This action applies multi-level tiling to improve cache locality by reducing
    the working set that must fit in L1/L2 cache. It targets the operation tagged
    with 'tag = "operation_0"' and tiles its iteration space with user-specified
    tile sizes.
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Define action parameters.
        
        Returns:
            dict: Parameter specifications with defaults.
                - tile_sizes (list[int]): Per-loop tile dimensions (0 = no tile).
        """
        return {
            "tile_sizes": {
                "type": "list[int]",
                "description": "Tile sizes for each loop dimension. 0 = skip tiling on that dimension.",
                "default": [32, 32, 32],  # Common L1-cache-friendly default for 3D loops
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if tiling is applicable.
        
        Verifies:
        1. The target operation (tag = "operation_0") exists in the code.
        2. tile_sizes is a valid non-empty list.
        3. At least one tile size is non-zero (not a no-op).
        
        Args:
            code (str): Input MLIR code.
            params (dict): Action parameters.
        
        Returns:
            bool: True if preconditions are met, False otherwise.
        """
        # Check that tag "operation_0" exists
        if 'tag = "operation_0"' not in code:
            return False
        
        # Validate tile_sizes parameter
        tile_sizes = params.get("tile_sizes", [])
        if not isinstance(tile_sizes, (list, tuple)):
            return False
        
        if len(tile_sizes) == 0:
            return False
        
        # Require at least one non-zero tile size (avoid no-op)
        if all(s == 0 for s in tile_sizes):
            return False
        
        # Ensure all tile sizes are integers >= 0
        try:
            if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
                return False
        except TypeError:
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Optional preprocessing (canonicalization).
        
        Currently a no-op; no preprocessing is required for tiling.
        
        Args:
            code (str): Input MLIR code.
            params (dict): Action parameters.
        
        Returns:
            str: Preprocessed code (identity in this case).
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Apply the tiling transformation.
        
        Constructs a Transform dialect snippet that:
        1. Matches the operation with tag "operation_0".
        2. Applies tile_using_for with the provided tile_sizes.
        3. Executes the transform.
        
        Args:
            code (str): Input MLIR code.
            params (dict): Action parameters.
        
        Returns:
            str: Transformed MLIR code, or original code if transform fails.
        """
        tile_sizes = params.get("tile_sizes", [])
        
        # Count non-zero tile sizes to determine number of output loops
        num_loops = len(tile_sizes)
        
        # Build the loop result tuple: (tiled_op, loop1, loop2, ..., loopN)
        loop_vars = ", ".join([f"%loop_{i}" for i in range(num_loops)])
        loop_types = ", ".join(["!transform.any_op"] * num_loops)
        
        # Build the transform module
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op_0 = transform.structured.match attributes{{tag = "operation_0"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op_0, {loop_vars} = transform.structured.tile_using_for %op_0 tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op, {loop_types})\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )
        
        try:
            transformed_code = run_transform_code(code, transform_code)
            return transformed_code
        except Exception:
            # On transform failure, return original code
            # (postcondition will reject it as a no-op)
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the transformation succeeded.
        
        Checks:
        1. The IR has been modified (not a no-op).
        2. The IR is still valid (non-empty, contains func.func).
        
        Args:
            before (str): Original MLIR code.
            after (str): Transformed MLIR code.
            params (dict): Action parameters.
        
        Returns:
            bool: True if transformation succeeded, False otherwise (no-op or invalid).
        """
        # Reject no-ops: IR must have changed
        if before.strip() == after.strip():
            return False
        
        # Minimal sanity check: IR should not be empty
        if not after or not after.strip():
            return False
        
        # Sanity: should still contain func.func (structure preserved)
        if "func.func" not in after:
            return False
        
        return True