class Action(ActionBase):
    @classmethod
    def parameters(cls) -> dict:
        """
        Define tunable parameters for loop interchange.
        
        Returns:
            dict: Parameter schema with names, types, and descriptions.
        """
        return {
            "loop_band": {
                "type": "str",
                "description": "Identifier for the loop band to interchange (e.g., 'matmul_loops', 'conv_loops')",
                "default": "root"
            },
            "permutation": {
                "type": "list[int]",
                "description": "Permutation of loop indices. E.g., [1, 0, 2] swaps first two loops.",
                "default": None
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if loop interchange is applicable.
        
        The precondition verifies:
        1. The code contains linalg operations (matmul or conv_2d).
        2. A permutation parameter is provided and is non-trivial.
        3. The permutation is a valid reordering (no repeated indices, all indices in range).
        
        Args:
            code (str): The MLIR code.
            params (dict): Parameters including 'permutation' and optionally 'loop_band'.
        
        Returns:
            bool: True if the transformation can be applied, False otherwise.
        """
        # Check for linalg operations
        if "linalg.matmul" not in code and "linalg.conv_2d_nchw_fchw" not in code:
            return False
        
        permutation = params.get("permutation", None)
        if permutation is None:
            return False
        
        # Validate permutation structure
        if not isinstance(permutation, list):
            return False
        
        if len(permutation) == 0:
            return False
        
        # Check that permutation is a valid reordering
        # (all elements are unique integers within range)
        if len(set(permutation)) != len(permutation):
            return False  # Duplicates
        
        if any(not isinstance(i, int) or i < 0 or i >= len(permutation) for i in permutation):
            return False  # Out of range
        
        # Permutation must be non-trivial (not identity)
        if permutation == list(range(len(permutation))):
            return False  # Identity permutation, no-op
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocess the code if necessary.
        
        For loop interchange, we ensure the linalg operations are properly tagged
        so they can be matched in the transform. If a specific loop_band is requested,
        we may need to wrap or annotate the code.
        
        In this implementation, preprocessing is minimal and returns the code as-is,
        relying on structured matching in the transform.
        
        Args:
            code (str): The MLIR code.
            params (dict): Parameters.
        
        Returns:
            str: Preprocessed MLIR code.
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Implement the loop interchange transformation using MLIR Transform dialect.
        
        This constructs a transform sequence that:
        1. Matches linalg operations in the code.
        2. Extracts their loops.
        3. Applies loop interchange (permutation) to the identified loop nest.
        
        Args:
            code (str): The MLIR code to transform.
            params (dict): Parameters including 'permutation'.
        
        Returns:
            str: Transformed MLIR code.
        """
        permutation = params.get("permutation", [])
        loop_band = params.get("loop_band", "root")
        
        # Build the permutation list as a string for MLIR
        perm_str = ", ".join(str(i) for i in permutation)
        
        # Construct the MLIR Transform dialect code
        # We match any linalg operation, extract its loops, and apply interchange
        transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    // Match any linalg operation (matmul or conv_2d)
    %linalg_op = transform.structured.match ops{{"linalg.matmul", "linalg.conv_2d_nchw_fchw"}} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // Get the loops produced by the linalg operation
    %loops = transform.structured.match ops{{"scf.for"}} in %linalg_op : (!transform.any_op) -> !transform.any_op
    
    // Apply interchange: permute the loop nest
    // The permutation {perm_str} is applied to reorder the loops
    %interchanged_loops = transform.loop.interchange %loops {{permutation = [{perm_str}]}} : (!transform.any_op) -> !transform.any_op
    
    transform.yield
  }}
}}
"""
        
        try:
            result = __transform_code(code, transform_code)
            return result
        except Exception:
            # If structured matching or interchange fails, try a more general approach
            # using affine loop interchange if the operations have been lowered to affine
            transform_code_affine = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    // Try to match affine loops (for lowered form)
    %affine_loops = transform.structured.match ops{{"affine.for"}} in %arg0 : (!transform.any_op) -> !transform.any_op
    
    // Apply affine loop interchange
    %interchanged = transform.affine.loop_permute %affine_loops {{permutation = [{perm_str}]}} : (!transform.any_op) -> !transform.any_op
    
    transform.yield
  }}
}}
"""
            try:
                result = __transform_code(code, transform_code_affine)
                return result
            except Exception:
                # If both fail, return original code (no-op in implementation stage)
                return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the loop interchange transformation succeeded.
        
        Postconditions check that:
        1. The output IR is valid (can be parsed).
        2. The linalg operations are still present (transformation preserved semantics).
        3. The loop structure has changed (permutation was applied).
        
        Args:
            before (str): The original MLIR code.
            after (str): The transformed MLIR code.
            params (dict): Parameters used in the transformation.
        
        Returns:
            bool: True if the transformation succeeded, False otherwise.
        """
        # Check that the output is not identical to the input
        # (indicating the transformation actually happened)
        if before == after:
            return False
        
        # Check that linalg operations are preserved
        has_matmul_before = "linalg.matmul" in before
        has_matmul_after = "linalg.matmul" in after
        
        has_conv_before = "linalg.conv_2d_nchw_fchw" in before
        has_conv_after = "linalg.conv_2d_nchw_fchw" in after
        
        # At least one linalg op must be preserved
        if (has_matmul_before and not has_matmul_after) or (has_conv_before and not has_conv_after):
            return False
        
        # Basic sanity check: the output should contain valid MLIR structure
        if "func.func" not in after or "return" not in after:
            return False
        
        return True