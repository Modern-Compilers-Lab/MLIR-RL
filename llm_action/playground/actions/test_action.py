from typing import Tuple

from llm_action.src.actions.base import ActionBase

def test_action(code: str, action: ActionBase, parameters: dict) -> Tuple[bool, str, bool]:
    """
    Test an action by checking precondition, applying implementation, and checking postcondition.
    Returns a tuple of (precondition_met, transformed_code, postcondition_met)
    """
    precondition_met = action.precondition(code, parameters)
    if not precondition_met:
        return False, code, False
    
    transformed_code = action.implement(code, parameters)
    postcondition_met = action.postcondition(code, transformed_code, parameters)
    
    return True, transformed_code, postcondition_met
