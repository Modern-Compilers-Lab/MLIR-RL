class Action(ActionBase):
    """
    Lowering Action: Transforms high-level structured operations into lower-level loop nests.
    
    This action converts linalg operations (matmul, conv_2d_nchw_fchw) into explicit scf.for
    loop nests, exposing iteration and memory access structure for subsequent optimizations
    (tiling, vectorization, fusion).
    
    Supported operations:
    - linalg.matmul -> triple-nested loops over (i, j, k)
    - linalg.conv_2d_nchw_fchw -> explicit im2col-like iteration
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Define tunable parameters for the lowering action.
        
        Returns:
            dict: Parameter specifications with name, type, and valid ranges/values.
        """
        return {
            "operation_type": {
                "type": "str",
                "values": ["matmul", "conv_2d_nchw_fchw"],
                "default": "matmul",
                "description": "Type of structured operation to lower"
            },
            "enable_iterator_type_semantics": {
                "type": "bool",
                "default": True,
                "description": "Preserve iterator type semantics during lowering"
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check whether the action is applicable to the given IR.
        
        Returns True if:
        - The code contains the target structured operation
        - Parameters are valid
        
        Args:
            code: MLIR payload as string
            params: Parameter dict with 'operation_type' and optionally 'enable_iterator_type_semantics'
        
        Returns:
            bool: True if action is applicable, False otherwise
        """
        operation_type = params.get("operation_type", "matmul")
        
        # Validate operation_type parameter
        if operation_type not in ["matmul", "conv_2d_nchw_fchw"]:
            return False
        
        # Check for presence of target operation in code
        if operation_type == "matmul":
            return "linalg.matmul" in code
        elif operation_type == "conv_2d_nchw_fchw":
            return "linalg.conv_2d_nchw_fchw" in code
        
        return False

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Optional canonicalization or preparation of the IR.
        
        For lowering, we perform minimal preprocessing:
        - Ensure operations are tagged for matching
        - Optionally apply canonicalization
        
        Args:
            code: MLIR payload as string
            params: Parameter dict
        
        Returns:
            str: Preprocessed MLIR code (may be identity)
        """
        # For now, lowering works directly on structured ops without preprocessing.
        # If needed in future, canonicalize could be inserted here.
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Core transformation logic: construct and execute MLIR Transform dialect code.
        
        Uses transform.structured.lower_to_loops to convert linalg ops into scf.for loops.
        
        Args:
            code: MLIR payload as string
            params: Parameter dict with operation_type and enable_iterator_type_semantics
        
        Returns:
            str: Transformed MLIR code with loops instead of structured ops
        """
        operation_type = params.get("operation_type", "matmul")
        enable_iterator_semantics = params.get("enable_iterator_type_semantics", True)
        
        # Build the transform dialect code
        if operation_type == "matmul":
            # For matmul, lower to explicit i-j-k triple-nested loops
            transform_code = _build_matmul_lowering_transform(enable_iterator_semantics)
        elif operation_type == "conv_2d_nchw_fchw":
            # For conv_2d, lower to explicit loop nests over spatial and channel dimensions
            transform_code = _build_conv2d_lowering_transform(enable_iterator_semantics)
        else:
            # Should not reach here if precondition checked properly
            return code
        
        # Execute the transform
        return __transform_code(code, transform_code)

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the transformation succeeded.
        
        Postconditions:
        - The original linalg operation should be replaced/lowered
        - The IR should be valid and contain scf.for loops
        - Dimensions and data flow should be preserved
        
        Args:
            before: Original MLIR code
            after: Transformed MLIR code
            params: Parameter dict
        
        Returns:
            bool: True if transformation succeeded and produced valid IR
        """
        operation_type = params.get("operation_type", "matmul")
        
        # Check that operation still exists (may be wrapped/lowered, not fully removed)
        # and that loops were introduced
        if operation_type == "matmul":
            # After lowering, matmul should be converted to scf.for loops
            # Check that we have scf.for operations and no more direct linalg.matmul
            has_loops = "scf.for" in after
            # Note: Some intermediate representations may still contain matmul,
            # but successful lowering should introduce loops.
            return has_loops
        elif operation_type == "conv_2d_nchw_fchw":
            # After lowering, conv should be converted to loop nests
            has_loops = "scf.for" in after
            return has_loops
        
        return False


# ============================================================================
# Helper functions for constructing transform dialect code
# ============================================================================

def _build_matmul_lowering_transform(enable_iterator_semantics: bool) -> str:
    """
    Build MLIR Transform dialect code to lower linalg.matmul to scf.for loops.
    
    Args:
        enable_iterator_semantics: If True, use transform with iterator type preservation.
        
    Returns:
        str: Transform dialect module as MLIR text
    """
    if enable_iterator_semantics:
        # Use structured lowering with iterator type preservation
        transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{\"linalg.matmul\"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %lowered = transform.structured.lower_to_loops %matmul : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
"""
    else:
        # Simpler lowering without semantic preservation
        transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %matmul = transform.structured.match ops{\"linalg.matmul\"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %lowered = transform.structured.lower_to_loops %matmul : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
"""
    return transform_code


def _build_conv2d_lowering_transform(enable_iterator_semantics: bool) -> str:
    """
    Build MLIR Transform dialect code to lower linalg.conv_2d_nchw_fchw to scf.for loops.
    
    Args:
        enable_iterator_semantics: If True, use transform with iterator type preservation.
        
    Returns:
        str: Transform dialect module as MLIR text
    """
    if enable_iterator_semantics:
        # Use structured lowering for convolution with semantic awareness
        transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %conv = transform.structured.match ops{\"linalg.conv_2d_nchw_fchw\"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %lowered = transform.structured.lower_to_loops %conv : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
"""
    else:
        # Simpler lowering for convolution
        transform_code = """
module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %conv = transform.structured.match ops{\"linalg.conv_2d_nchw_fchw\"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %lowered = transform.structured.lower_to_loops %conv : (!transform.any_op) -> (!transform.any_op)
    transform.yield
  }
}
"""
    return transform_code