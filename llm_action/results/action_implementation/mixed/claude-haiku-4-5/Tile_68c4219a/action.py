from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class Tile(ActionBase):
    """
    Tiling action: Partition loop nests into smaller rectangular blocks
    such that intermediate results fit in L1/L2 cache.
    
    Uses MLIR Transform dialect's transform.structured.tile_using_for
    to generate nested scf.for loops with tensor slicing operations.
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Returns the parameter schema for the tiling action.
        
        Parameters:
        - tile_sizes: list of integers representing tile sizes for each loop dimension.
                      0 means no tiling for that dimension.
        """
        return {
            "tile_sizes": {
                "type": "list",
                "description": "Tile sizes for each loop dimension (0 = no tiling)"
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check whether the action is applicable to the given IR.
        
        Verifies:
        1. The code contains the tagged operation: tag = "operation_0"
        2. tile_sizes parameter exists and is a non-empty list
        3. All tile sizes are non-negative integers
        4. Not all tile sizes are zero (would be a no-op)
        
        Returns False if any precondition fails.
        """
        # Check for tag presence
        if 'tag = "operation_0"' not in code:
            return False
        
        # Check parameters
        if "tile_sizes" not in params:
            return False
        
        tile_sizes = params.get("tile_sizes")
        
        # Validate tile_sizes is a list
        if not isinstance(tile_sizes, (list, tuple)):
            return False
        
        # Must have at least one tile size
        if len(tile_sizes) == 0:
            return False
        
        # All tile sizes must be non-negative integers
        try:
            tile_sizes_int = [int(s) for s in tile_sizes]
            if any(s < 0 for s in tile_sizes_int):
                return False
        except (ValueError, TypeError):
            return False
        
        # Reject if all tile sizes are zero (no-op)
        if all(s == 0 for s in tile_sizes_int):
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Optional preprocessing step. Identity function for tiling.
        
        No canonicalization or tag manipulation is performed.
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Apply the tiling transformation using MLIR Transform dialect.
        
        Constructs a transform.structured.tile_using_for operation that:
        1. Matches the operation with tag = "operation_0"
        2. Tiles it with the specified tile_sizes
        3. Returns the tiled op and generated loops
        
        Handles the return unpacking based on the number of non-zero tile sizes.
        """
        tile_sizes = params.get("tile_sizes")
        
        # Convert to list of ints if needed
        tile_sizes_list = [int(s) for s in tile_sizes]
        
        # Count non-zero tile sizes to determine result arity
        non_zero_count = sum(1 for s in tile_sizes_list if s != 0)
        
        # Build the tile_sizes attribute string
        tile_sizes_str = str(tile_sizes_list).replace('[', '').replace(']', '')
        
        # Build the result unpacking pattern
        # Format: %tiled, %loops:N where N is the number of non-zero tile sizes
        if non_zero_count == 0:
            # This should be caught by precondition, but handle gracefully
            return code
        
        result_types = ', '.join(['!transform.any_op'] * (non_zero_count + 1))
        loop_result_part = ', '.join(['!transform.any_op'] * non_zero_count)
        result_unpack = f'%tiled, %loops:{non_zero_count}' if non_zero_count > 0 else '%tiled'
        
        # Construct the transform IR
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %op = transform.structured.match attributes{{tag = "operation_0"}} in %arg0\n'
            f'      : (!transform.any_op) -> !transform.any_op\n'
            f'    {result_unpack} = transform.structured.tile_using_for %op\n'
            f'      tile_sizes [{tile_sizes_str}]\n'
            f'      : (!transform.any_op) -> ({result_types})\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )
        
        # Apply the transform
        try:
            transformed = run_transform_code(code, transform_code)
            return transformed
        except Exception:
            # If transform fails, return original code
            # Postcondition will detect no-op and return False
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the transformation was successful.
        
        Checks:
        1. The IR changed (before != after)
        2. The result contains a func.func (basic sanity)
        3. The tagged operation still exists (tag = "operation_0")
        
        Returns False if any postcondition fails.
        """
        # Reject no-op: if IR is identical, transform failed
        if before.strip() == after.strip():
            return False
        
        # Basic sanity: must still contain a function
        if 'func.func' not in after:
            return False
        
        # Verify the tagged operation still exists
        if 'tag = "operation_0"' not in after:
            return False
        
        # Additional sanity: must contain scf.for loops (tiling should generate them)
        if 'scf.for' not in after and all(int(s) != 0 for s in params.get("tile_sizes", [])):
            # Only require scf.for if at least one dimension was tiled
            return False
        
        return True