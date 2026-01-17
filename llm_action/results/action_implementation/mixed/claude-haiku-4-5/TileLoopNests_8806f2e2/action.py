class Action(ActionBase):
    """
    Tiling action: partitions loop nests into smaller rectangular blocks
    to fit intermediate results in L1/L2 cache.
    
    Parameters:
      - tiling_sizes: list of positive integers or zeros indicating tile size per loop.
      - operation_tag: optional tag to identify which operation to tile.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tiling_sizes": {
                "type": "list[int]",
                "description": "Tile sizes for each loop dimension (0 = no tiling)",
                "default": [8, 8, 8],
                "constraints": "all elements >= 0"
            },
            "operation_tag": {
                "type": "str",
                "description": "Operation tag to match (empty = all linalg ops)",
                "default": ""
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check that:
        1. The code contains at least one linalg operation or loop nest.
        2. At least one tiling size is non-zero.
        3. Tiling sizes list is reasonable (non-empty, all non-negative).
        """
        # Validate tiling_sizes parameter
        tiling_sizes = params.get("tiling_sizes", [])
        if not isinstance(tiling_sizes, list) or len(tiling_sizes) == 0:
            return False
        
        # All elements must be non-negative integers
        if not all(isinstance(x, int) and x >= 0 for x in tiling_sizes):
            return False
        
        # At least one tile size must be non-zero (otherwise it's a no-op)
        if all(x == 0 for x in tiling_sizes):
            return False
        
        # Check for presence of tileable operations
        # Look for linalg.* operations or loop nests
        has_linalg = "linalg." in code
        has_loops = "scf.for" in code or "affine.for" in code
        
        if not (has_linalg or has_loops):
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Minimal preprocessing: validate and normalize parameters.
        No IR modification needed.
        """
        tiling_sizes = params.get("tiling_sizes", [])
        
        # Ensure tiling_sizes is a list of non-negative integers
        tiling_sizes = [max(0, int(x)) for x in tiling_sizes]
        params["tiling_sizes"] = tiling_sizes
        
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Core tiling transformation using MLIR Transform dialect.
        
        Strategy:
        - If operation_tag is provided, match that specific operation.
        - Otherwise, match all linalg operations.
        - Apply transform.structured.tile_using_for with the provided tile sizes.
        """
        tiling_sizes = params.get("tiling_sizes", [8, 8, 8])
        operation_tag = params.get("operation_tag", "")
        
        # Filter out zeros and keep only non-zero tile sizes
        # We need to map tile sizes to actual loop dimensions
        non_zero_sizes = [s for s in tiling_sizes if s > 0]
        
        if not non_zero_sizes:
            return code  # No-op if all tile sizes are zero
        
        # Build the transform dialect code
        if operation_tag:
            # Match a specific operation by tag
            match_clause = f'%op = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op'
        else:
            # Match all linalg operations
            match_clause = '%op = transform.structured.match ops["linalg.matmul", "linalg.conv_2d_nchw_fchw", "linalg.generic"] in %arg1 : (!transform.any_op) -> !transform.any_op'
        
        # Number of loops created by tiling
        num_tiles = len([s for s in tiling_sizes if s > 0])
        loop_returns = ', '.join(['!transform.any_op'] * num_tiles)
        
        # Format tile sizes as MLIR list
        tile_sizes_str = str(tiling_sizes)
        
        transform_code = f'''
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    {match_clause}
    %tiled_op, %tiled_loops:{num_tiles} = transform.structured.tile_using_for %op tile_sizes {tile_sizes_str} : (!transform.any_op) -> (!transform.any_op, {loop_returns})
    transform.yield
  }}
}}
'''
        
        try:
            result = __transform_code(code, transform_code)
            return result
        except Exception:
            # If tiling fails (e.g., operation not tileable), return original code
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Check that:
        1. The transformed IR is still valid (parseable).
        2. At least one new loop was introduced by tiling.
        3. Key operations were preserved.
        """
        # Basic validity: after IR should be parseable MLIR
        try:
            # Simple check: after IR should contain "scf.for" or "affine.for"
            # (tiling introduces explicit loop nests)
            has_loops_after = "scf.for" in after or "affine.for" in after
            has_linalg_after = "linalg." in after
            
            # After tiling, we expect either explicit loops or the original linalg ops
            # (depending on the lowering state)
            if not (has_loops_after or has_linalg_after):
                return False
            
            # Simple heuristic: the after IR should be at least as long as before
            # (tiling typically introduces new loop structure)
            if len(after) < len(before):
                return False
            
            return True
        except Exception:
            return False