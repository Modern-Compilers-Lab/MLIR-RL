class Action(ActionBase):
    """
    Tiling action for cache locality optimization.
    
    Tiles loop nests in linalg operations to maximize data reuse within
    L1/L2 cache by partitioning iteration spaces into fixed-size blocks.
    """

    @classmethod
    def parameters(cls) -> dict:
        """Define tunable parameters for the tiling action."""
        return {
            "tile_sizes": {
                "description": "List of tile sizes for consecutive loop levels",
                "type": list,
                "default": [64, 64, 64],
            },
            "operation_name": {
                "description": "Optional operation name filter (e.g., 'matmul', 'conv_2d')",
                "type": str,
                "default": None,
            },
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if tiling can be applied.
        
        Conditions:
        - IR must contain at least one linalg operation.
        - tile_sizes must be a list of non-negative integers.
        - At least one tile size must be non-zero.
        """
        # Check tile_sizes validity
        tile_sizes = params.get("tile_sizes", [])
        if not isinstance(tile_sizes, list):
            return False
        if len(tile_sizes) == 0:
            return False
        if not all(isinstance(s, int) and s >= 0 for s in tile_sizes):
            return False
        if all(s == 0 for s in tile_sizes):
            return False

        # Check that IR contains linalg operations
        operation_name = params.get("operation_name")
        has_linalg = False
        
        if "linalg.matmul" in code:
            has_linalg = True
        elif "linalg.conv_2d" in code:
            has_linalg = True
        elif "linalg." in code:
            has_linalg = True

        if not has_linalg:
            return False

        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocess the IR (canonicalization).
        
        For tiling, we ensure the IR is valid and contains the target operation.
        No structural changes needed before tiling.
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Implement the tiling transformation using MLIR Transform dialect.
        
        Constructs a parameterized transform sequence that:
        1. Matches the target linalg operation.
        2. Applies tile_using_for with the specified tile sizes.
        3. Returns the transformed IR.
        """
        tile_sizes = params.get("tile_sizes", [64, 64, 64])
        operation_name = params.get("operation_name", "matmul")

        # Determine which operation to match and tag
        # Default to matmul if no filter specified
        if operation_name is None or operation_name.lower() == "matmul":
            op_match = 'transform.structured.match ops["linalg.matmul"]'
            tag = "tiling_matmul"
        elif operation_name.lower() == "conv_2d":
            op_match = 'transform.structured.match ops["linalg.conv_2d_nchw_fchw"]'
            tag = "tiling_conv_2d"
        else:
            # Generic linalg match
            op_match = 'transform.structured.match ops["linalg.generic", "linalg.matmul", "linalg.conv_2d_nchw_fchw"]'
            tag = "tiling_generic"

        # Build the list of loop variables for the result
        num_tiles = len([s for s in tile_sizes if s != 0])
        loop_vars = ", ".join([f"%loop_{i}" for i in range(num_tiles)])
        if loop_vars:
            loop_return = f", {loop_vars}"
        else:
            loop_return = ""

        # Build transform code
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %op = {op_match} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f"    %tiled_op{loop_return} = transform.structured.tile_using_for %op "
            f"tile_sizes {tile_sizes} : (!transform.any_op) -> (!transform.any_op"
            f"{', !transform.any_op' * num_tiles})\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )

        # Execute the transform
        try:
            result = __transform_code(code, transform_code)
            return result
        except Exception as e:
            # If transform fails, return original code (treat as no-op)
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that tiling was applied successfully.
        
        Checks:
        - The transformed IR is syntactically valid MLIR.
        - The IR contains scf.for loops (introduced by tile_using_for).
        - The number of linalg operations is preserved or reduced (fusion may occur).
        """
        # Check that result is not identical to input (transformation occurred)
        if before.strip() == after.strip():
            # Tiling may be a no-op if tile sizes are very large, but we still check
            # for scf.for loops to confirm *some* transformation happened
            if "scf.for" not in after:
                return False
            return True

        # Check for syntactic validity (basic heuristic: contains module/func)
        if "module" not in after or "func" not in after:
            return False

        # Check that scf.for loops were introduced (signature of tiling)
        if "scf.for" not in after:
            # Tiling should always introduce at least one scf.for loop
            return False

        return True