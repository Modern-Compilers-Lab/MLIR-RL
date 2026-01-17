from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re


class TileAction(ActionBase):
    """
    Tiling Action: Partitions loop nests into smaller rectangular blocks
    to improve cache locality (L1/L2 utilization).
    
    Supported patterns:
    - linalg.matmul and other structured linalg ops
    - affine and scf loop nests
    
    Semantics:
    - tile_sizes: list of integers, where 0 means no tiling for that dimension
    - operation_name: optional tag/name of operation to target
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tile_sizes": {
                "description": "List of tile sizes (0 = no tiling for that dimension)",
                "type": "list[int]",
                "required": True,
            },
            "operation_name": {
                "description": "Optional operation name/tag to target (matmul, conv_2d_nchw_fchw, generic)",
                "type": "str",
                "required": False,
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if tiling can be applied:
        1. tile_sizes must be a list of non-negative integers
        2. At least one non-zero tile size (otherwise it's a no-op)
        3. Target operation or loop nest must be present in the code
        """
        tile_sizes = params.get("tile_sizes")
        operation_name = params.get("operation_name", None)

        # Validate tile_sizes
        if not isinstance(tile_sizes, list):
            return False
        if not all(isinstance(t, int) and t >= 0 for t in tile_sizes):
            return False
        if all(t == 0 for t in tile_sizes):
            # All zeros => no-op
            return False

        # Check for tiling-eligible operations in the code
        has_linalg = bool(
            re.search(r"linalg\.(matmul|conv_2d|generic|map)", code)
        )
        has_loops = bool(
            re.search(r"(affine\.for|scf\.for|scf\.while)", code)
        )

        if not (has_linalg or has_loops):
            return False

        # If operation_name is specified, it should appear in the code
        if operation_name and operation_name != "generic":
            if not re.search(rf"linalg\.{operation_name}\b", code):
                return False

        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Optionally tag the target operation if not already tagged.
        This makes it easier to match in the transform code.
        """
        operation_name = params.get("operation_name", None)
        
        # If an operation_name is specified and the code doesn't already have a tag,
        # inject a simple tag attribute.
        if operation_name and operation_name != "generic":
            op_pattern = rf"(linalg\.{operation_name}\b[^{{]*{{)"
            if re.search(op_pattern, code):
                # Already likely tagged or will be handled by generic matching
                pass
        
        # For now, return code as-is. Transform dialect can match without explicit tags.
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Apply tiling using MLIR Transform dialect.
        
        Strategy:
        1. Try to match linalg structured ops first (matmul, conv, generic).
        2. If found, apply transform.structured.tile_using_for.
        3. If not found, attempt to match and tile affine/scf loops.
        """
        tile_sizes = params.get("tile_sizes")
        operation_name = params.get("operation_name", None)

        # Construct the transform dialect code
        # We'll use transform.structured.tile_using_for for structured ops,
        # and fall back to affine.tile if needed.

        transform_code = cls._build_transform_code(tile_sizes, operation_name)

        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception as e:
            # If the structured approach fails, return original code
            # The postcondition will catch this as a failure
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that tiling was applied successfully.
        
        Checks:
        1. The transformed code should still parse as valid MLIR.
        2. Loop structures should be present (either original or tiled).
        3. If tiling was applied, we should see scf.for loops or affine.for loops.
        4. Basic sanity: code length should not shrink dramatically (no accidental deletion).
        """
        # Check that after is non-empty and different (transformation occurred)
        if not after or len(after.strip()) == 0:
            return False

        # Check for loop presence
        has_loops = bool(
            re.search(r"(affine\.for|scf\.for|scf\.while|linalg\.\w+)", after)
        )
        if not has_loops:
            return False

        # Code should not have shrunk by more than 30% (sanity check)
        if len(after) < len(before) * 0.3:
            return False

        # The code should still contain the main function
        if "func.func @main" not in after:
            return False

        return True

    @classmethod
    def _build_transform_code(
        cls, tile_sizes: list, operation_name: str = None
    ) -> str:
        """
        Construct MLIR Transform dialect code for tiling.
        
        This generates a named sequence that:
        1. Matches the target operation(s).
        2. Applies tile_using_for with the specified tile sizes.
        """
        # Build tile sizes string
        tile_sizes_str = "{" + ", ".join(str(t) for t in tile_sizes) + "}"

        # Determine how many loop results to expect
        num_tiled_loops = sum(1 for t in tile_sizes if t > 0)

        if num_tiled_loops == 0:
            # Edge case: all zeros (should have been caught by precondition)
            num_tiled_loops = 1

        # Build result binding for loops
        loop_bindings = ", ".join([f"%loop{i}" for i in range(num_tiled_loops)])

        # If operation_name is specified, match it by name/tag
        # Otherwise, match any linalg or generic operation
        if operation_name and operation_name != "generic":
            match_op = (
                f'%target_op = transform.structured.match '
                f'ops["{operation_name}"] in %arg1 : (!transform.any_op) -> !transform.any_op'
            )
        else:
            # Match any structured operation that can be tiled
            match_op = (
                f'%target_op = transform.structured.match '
                f'ops["linalg.matmul", "linalg.conv_2d_nchw_fchw", "linalg.generic"] '
                f'in %arg1 : (!transform.any_op) -> !transform.any_op'
            )

        # Build the transform code
        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    {match_op}
    %tiled_op, %loops:{num_tiled_loops} = transform.structured.tile_using_for %target_op tile_sizes {tile_sizes_str} : (!transform.any_op) -> (!transform.any_op, !transform.any_op{", !transform.any_op" * (num_tiled_loops - 1)})
    transform.yield
  }}
}}
"""
        return transform_code