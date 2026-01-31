from llm_action.src.actions.base import ActionBase
from llm_action.src.utils.transformation import run_transform_code


class LoopInterchange(ActionBase):
    """
    Loop Interchange action that reorders loop nesting on linalg operations.
    
    This action generalizes named linalg operations to linalg.generic, then applies
    the specified iterator interchange permutation to reorder loop nesting.
    """

    @classmethod
    def parameters(cls) -> dict:
        """
        Returns the parameter specification for loop interchange.
        
        Returns:
            dict: Parameter specification with 'iterator_interchange' key.
        """
        return {
            "iterator_interchange": {
                "type": "list[int]",
                "description": "A permutation array specifying the new order of loop iterators.",
                "required": True
            }
        }

    @classmethod
    def precondition(cls, code: str, params: dict) -> bool:
        """
        Checks if loop interchange is applicable to the given code.
        
        Preconditions:
        1. The code must contain the tag 'operation_0'
        2. iterator_interchange must be present and be a list
        3. iterator_interchange must not be empty
        4. iterator_interchange must contain unique integers (valid permutation)
        5. iterator_interchange must not be the identity permutation
        
        Args:
            code (str): The MLIR code to check.
            params (dict): Parameters containing 'iterator_interchange'.
        
        Returns:
            bool: True if applicable, False otherwise.
        """
        # Check for operation_0 tag
        if 'tag = "operation_0"' not in code:
            return False
        
        # Check iterator_interchange parameter exists and is valid
        if "iterator_interchange" not in params:
            return False
        
        interchange = params["iterator_interchange"]
        
        # Must be a list
        if not isinstance(interchange, list):
            return False
        
        # Must not be empty
        if len(interchange) == 0:
            return False
        
        # Must contain only integers
        if not all(isinstance(i, int) for i in interchange):
            return False
        
        # Must be non-negative
        if not all(i >= 0 for i in interchange):
            return False
        
        # Must be a valid permutation (no duplicates, contiguous from 0)
        if len(set(interchange)) != len(interchange):
            return False
        
        if set(interchange) != set(range(len(interchange))):
            return False
        
        # Must not be identity permutation (would be a no-op)
        if interchange == list(range(len(interchange))):
            return False
        
        return True

    @classmethod
    def preprocess(cls, code: str, params: dict) -> str:
        """
        Preprocessing step (identity for this action).
        
        Args:
            code (str): The MLIR code.
            params (dict): Parameters.
        
        Returns:
            str: Unmodified code.
        """
        return code

    @classmethod
    def implement(cls, code: str, params: dict) -> str:
        """
        Implements loop interchange using MLIR Transform dialect.
        
        Strategy:
        1. Match the operation with tag "operation_0"
        2. Generalize named operations to linalg.generic (required for interchange)
        3. Apply iterator interchange with the specified permutation
        
        Args:
            code (str): The MLIR code to transform.
            params (dict): Parameters containing 'iterator_interchange'.
        
        Returns:
            str: Transformed MLIR code, or original code if transformation fails.
        """
        interchange = params["iterator_interchange"]
        
        # Format the permutation as a dense array attribute
        permutation_str = "[" + ", ".join(str(i) for i in interchange) + "]"
        
        # Construct the transform IR
        transform_code = (
            f'module attributes {{transform.with_named_sequence}} {{\n'
            f'  transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{\n'
            f'    %matched = transform.structured.match attributes {{tag = "operation_0"}} in %arg0 '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %generalized = transform.structured.generalize %matched '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    %interchanged = transform.structured.interchange %generalized '
            f'iterator_interchange = {permutation_str} '
            f': (!transform.any_op) -> !transform.any_op\n'
            f'    transform.yield\n'
            f'  }}\n'
            f'}}\n'
        )
        
        try:
            return run_transform_code(code, transform_code)
        except Exception:
            # If transformation fails, return original code
            # Postcondition will detect this as a no-op
            return code

    @classmethod
    def postcondition(cls, before: str, after: str, params: dict) -> bool:
        """
        Verifies that loop interchange was successful.
        
        Postconditions:
        1. The code must have changed (not a no-op)
        2. The transformed code must be non-empty
        3. The transformed code must still contain a function definition
        
        Args:
            before (str): Original MLIR code.
            after (str): Transformed MLIR code.
            params (dict): Parameters.
        
        Returns:
            bool: True if transformation succeeded, False otherwise.
        """
        # Check for no-op (code unchanged)
        if after.strip() == before.strip():
            return False
        
        # Check that result is non-empty
        if not after or not after.strip():
            return False
        
        # Check that basic structure is preserved (contains func.func)
        if "func.func" not in after:
            return False
        
        return True