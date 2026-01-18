from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code
import re


class LoopInterchange(ActionBase):
    """
    Loop Interchange transformation using MLIR Transform dialect.
    
    Reorders loop dimensions in structured operations (linalg.generic, etc.) to improve
    memory access patterns and enable better vectorization. Targets the operation tagged
    with 'tag = "operation_0"'.
    
    Parameters:
    - iterator_permutation: A list of integers specifying the new loop order.
      Example: [1, 0, 2] swaps loops 0 and 1, keeps loop 2 in place.
    """
    
    @classmethod
    def parameters(cls) -> dict:
        return {
            "iterator_permutation": {
                "type": "list[int]",
                "description": "Permutation of iterator indices for loop reordering",
                "default": [1, 0]
            }
        }
    
    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Check if the transformation is applicable:
        1. Code must contain the target operation with tag = "operation_0"
        2. The operation must support iterator interchange (e.g., linalg.generic)
        3. The permutation must be valid (non-negative integers, well-formed)
        """
        # Check if tag exists
        if 'tag = "operation_0"' not in code:
            return False
        
        # Extract permutation
        perm = params.get("iterator_permutation", [1, 0])
        
        # Validate permutation is a list of non-negative integers
        if not isinstance(perm, list) or len(perm) == 0:
            return False
        
        if not all(isinstance(i, int) and i >= 0 for i in perm):
            return False
        
        # Check that permutation is not a no-op (identity)
        if perm == list(range(len(perm))):
            return False
        
        # Basic IR structure check
        if "func.func @main" not in code:
            return False
        
        return True
    
    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        No preprocessing required; the code is used as-is.
        """
        return code
    
    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Apply loop interchange via MLIR Transform dialect.
        
        Constructs a transform sequence that:
        1. Matches the operation with tag = "operation_0"
        2. Applies structured.interchange with the specified permutation
        3. Returns the transformed code
        """
        perm = params.get("iterator_permutation", [1, 0])
        
        # Format the permutation as a dense array
        perm_str = ", ".join(str(i) for i in perm)
        
        # Build the transform module
        transform_code = (
            f"module attributes {{transform.with_named_sequence}} {{\n"
            f"  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n"
            f'    %target = transform.structured.match attributes{{tag = "operation_0"}} in %arg0 : (!transform.any_op) -> !transform.any_op\n'
            f"    %interchanged = transform.structured.interchange %target iterator_interchange = [{perm_str}] : (!transform.any_op) -> !transform.any_op\n"
            f"    transform.yield\n"
            f"  }}\n"
            f"}}\n"
        )
        
        # Apply the transformation
        try:
            transformed = run_transform_code(code, transform_code)
            return transformed
        except Exception:
            # On failure, return original code; postcondition will catch this
            return code
    
    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verify the transformation succeeded:
        1. The code must have changed (not a no-op)
        2. The result must be valid MLIR (non-empty, still contains func.func)
        3. The tag must still be present
        """
        # No-op check: if code unchanged, transformation failed
        if before.strip() == after.strip():
            return False
        
        # Basic IR validity checks
        if not after.strip():
            return False
        
        if "func.func" not in after:
            return False
        
        # Tag should still be present
        if 'tag = "operation_0"' not in after:
            return False
        
        return True