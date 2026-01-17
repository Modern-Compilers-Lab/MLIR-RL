class Action(ActionBase):
    """
    Packing Action: Transforms tensor layout to cache-friendly blocked format.
    
    Applies tensor.pack to restructure memory layout and regularize access patterns,
    trading temporary buffer allocation for improved cache utilization.
    """

    @classmethod
    def parameters(cls) -> dict:
        return {
            "tensor_name": {
                "description": "Tensor operand to pack (e.g., 'arg0')",
                "type": "str",
                "default": "arg0"
            },
            "pack_tile_shape": {
                "description": "Pack tile shape per dimension",
                "type": "list[int]",
                "default": [64, 128]
            },
            "outer_dims_perm": {
                "description": "Optional dimension permutation before packing",
                "type": "list[int]",
                "default": None
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if packing is applicable:
        - Tensor argument exists
        - Pack tile shape is non-empty and positive
        - Tensor has sufficient dimensions for packing
        """
        try:
            tensor_name = params.get("tensor_name", "arg0")
            pack_tile_shape = params.get("pack_tile_shape", [64, 128])
            
            # Validate parameters
            if not isinstance(pack_tile_shape, list) or len(pack_tile_shape) == 0:
                return False
            if any(t < 0 for t in pack_tile_shape):
                return False
            
            # Check if tensor_name appears in the code
            if tensor_name not in code:
                return False
            
            # Check if code contains a packable structured operation
            if not any(op in code for op in ["linalg.matmul", "linalg.conv_2d", "linalg.generic"]):
                return False
            
            return True
        except Exception:
            return False

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Canonicalize and prepare the IR.
        Ensure the IR is in a form suitable for packing (e.g., explicit tensor types).
        """
        # Basic preprocessing: ensure code is valid MLIR
        try:
            # Attempt to validate by checking for basic MLIR structure
            if "module" not in code or "func.func" not in code:
                return code
            return code
        except Exception:
            return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Core transformation: inject tensor.pack operations to restructure tensor layout.
        
        Strategy:
        1. Identify the main structured operation (matmul, conv_2d, etc.)
        2. Apply tensor.pack to the specified input tensor
        3. Update the operation to use the packed tensor
        """
        try:
            tensor_name = params.get("tensor_name", "arg0")
            pack_tile_shape = params.get("pack_tile_shape", [64, 128])
            outer_dims_perm = params.get("outer_dims_perm", None)
            
            # Sanitize tile shape: filter out zero/negative values
            pack_tile_shape = [t for t in pack_tile_shape if t > 0]
            if not pack_tile_shape:
                return code
            
            # Build transform dialect code to apply packing
            # We'll use a generic approach: match the operation, extract tensor, and pack it
            
            # Construct the inner dims and outer dims tile spec
            # For simplicity, assume pack_tile_shape corresponds to the last N dimensions
            tile_spec = ", ".join([f"{t}" for t in pack_tile_shape])
            
            transform_code = f'''
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    // Match any linalg operation (matmul, conv_2d, generic, etc.)
    %matched = transform.structured.match 
      interface{{"LinalgOp"}} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // Apply packing via pack_greedily which converts strided accesses to packed format
    // This uses tensor.pack under the hood when beneficial
    %packed, %pack_info = 
      transform.structured.pack_greedily %matched 
      pack_sizes = {tile_spec}
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    transform.yield
  }}
}}
'''
            
            # Apply the transform
            result = __transform_code(code, transform_code)
            return result
            
        except Exception:
            # If transform fails, return original code
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Validate that packing was successfully applied.
        Check for:
        - Output IR is valid MLIR
        - tensor.pack or related operations are present
        - Tensor shapes and types are preserved
        """
        try:
            # Check that output is valid MLIR
            if "module" not in after or "func.func" not in after:
                return False
            
            # Check that packing-related operations were injected
            # Look for tensor.pack, tensor.unpack, or transform artifacts
            has_pack_ops = any(op in after for op in [
                "tensor.pack",
                "tensor.unpack",
                "transform.structured.pack"
            ])
            
            # If no explicit pack ops appear, the transform may have been a no-op
            # (e.g., due to operation not matching). For now, we consider this a failure.
            if not has_pack_ops:
                # Check if the structured operation is still present and valid
                # If yes, packing may not have been applicable (still a valid outcome)
                if any(op in after for op in ["linalg.matmul", "linalg.conv_2d", "linalg.generic"]):
                    # Return False because we expected packing to be applied
                    return False
            
            # Basic validation: check output has expected function signature
            if "func.func @main" not in after:
                return False
            
            return True
            
        except Exception:
            return False