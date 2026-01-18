from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re


class LoopPeeling(ActionBase):
    """
    Loop Peeling Action: Separates loop boundary conditions and edge cases
    from the main loop body by peeling off the first or last iteration(s).
    
    This transformation enables optimization of the main loop without
    conditional overhead, supporting aggressive transformations like
    vectorization and loop unrolling.
    
    Works by:
    1. Matching the operation tagged with "operation_0"
    2. Converting it to scf.for loops (if not already lowered)
    3. Applying loop peeling to isolate boundary iterations
    """
    
    @classmethod
    def parameters(cls) -> dict:
        """
        Define tunable parameters for loop peeling.
        
        Returns:
            dict: Parameter specification with defaults
        """
        return {
            "peel_front": {
                "description": "Peel first (true) or last (false) iteration(s)",
                "type": "bool",
                "default": True
            },
            "fail_if_already_divisible": {
                "description": "Fail if loop already evenly divisible",
                "type": "bool",
                "default": False
            }
        }
    
    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if loop peeling is applicable to the given code.
        
        Preconditions:
        - Code must contain the tag "operation_0"
        - Must be a valid MLIR module
        - Parameters must be boolean values
        
        Args:
            code: The MLIR code to check
            params: Transformation parameters
            
        Returns:
            bool: True if transformation can be applied, False otherwise
        """
        # Check that tag "operation_0" exists
        if 'tag = "operation_0"' not in code:
            return False
        
        # Validate parameters are boolean
        peel_front = params.get("peel_front", True)
        fail_if_already_divisible = params.get("fail_if_already_divisible", False)
        
        if not isinstance(peel_front, bool) or not isinstance(fail_if_already_divisible, bool):
            return False
        
        # Check that code is valid MLIR (basic syntax check)
        if "module" not in code or "func.func" not in code:
            return False
        
        return True
    
    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocess the code if needed.
        
        Currently a no-op - no special preprocessing required.
        
        Args:
            code: The MLIR code
            params: Transformation parameters
            
        Returns:
            str: Preprocessed code (identity for this transformation)
        """
        return code
    
    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Apply loop peeling transformation to the code.
        
        Strategy:
        1. Match the operation with tag "operation_0"
        2. Attempt to lower to scf.for loops
        3. Apply loop.peel with specified parameters
        4. Return transformed code
        
        Args:
            code: The MLIR code to transform
            params: Transformation parameters (peel_front, fail_if_already_divisible)
            
        Returns:
            str: Transformed MLIR code
        """
        peel_front = params.get("peel_front", True)
        fail_if_already_divisible = params.get("fail_if_already_divisible", False)
        
        # Construct Transform dialect code to apply loop peeling
        transform_code = f'''module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    // Match the operation tagged with "operation_0"
    %matched = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 
      : (!transform.any_op) -> !transform.any_op
    
    // Strategy 1: Try direct conversion to loops
    // This works for linalg operations that implement TilingInterface
    %loops = transform.structured.convert_to_loops %matched 
      : (!transform.any_op) -> !transform.any_op
    
    // Apply loop peeling to the generated loops
    %peeled_loop, %remainder_loop = transform.loop.peel %loops 
      {{ peel_front = {str(peel_front).lower()}, fail_if_already_divisible = {str(fail_if_already_divisible).lower()} }} 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    transform.yield
  }}
}}
'''
        
        try:
            result = run_transform_code(code, transform_code)
            return result
        except Exception:
            # If direct conversion fails, try alternative pathway: tile then forall_to_for
            # This provides a fallback for cases where convert_to_loops isn't supported
            transform_code_fallback = f'''module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
    // Match the operation tagged with "operation_0"
    %matched = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 
      : (!transform.any_op) -> !transform.any_op
    
    // Fallback strategy: Tile first, then convert to for loops
    // Use minimal tile sizes to create loop structure
    %tiled, %forall = transform.structured.tile_using_forall %matched tile_sizes [0, 0] 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    // Convert scf.forall to scf.for loops
    %for_loops = transform.loop.forall_to_for %forall 
      : (!transform.any_op) -> !transform.any_op
    
    // Apply loop peeling to innermost loop structure
    %peeled_loop, %remainder_loop = transform.loop.peel %for_loops 
      {{ peel_front = {str(peel_front).lower()}, fail_if_already_divisible = {str(fail_if_already_divisible).lower()} }} 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    
    transform.yield
  }}
}}
'''
            try:
                result = run_transform_code(code, transform_code_fallback)
                return result
            except Exception:
                # If both strategies fail, return original code
                # The postcondition will detect this as a no-op
                return code
    
    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify that the loop peeling transformation succeeded.
        
        Postconditions:
        - The transformed code must differ from the original
        - The module structure must be preserved
        - func.func @main must still exist
        
        Args:
            before: Original MLIR code
            after: Transformed MLIR code
            params: Transformation parameters
            
        Returns:
            bool: True if transformation succeeded, False if it's a no-op or failed
        """
        # Check for actual code change (not a no-op)
        if before.strip() == after.strip():
            return False
        
        # Verify that essential structure is preserved
        if "func.func @main" not in after:
            return False
        
        if "module" not in after:
            return False
        
        # Verify that the transformation didn't break the operation tag
        # (the tag might be moved to the peeled/remainder loops, but should still exist)
        if "tag = " not in after and "operation_0" not in after:
            return False
        
        return True