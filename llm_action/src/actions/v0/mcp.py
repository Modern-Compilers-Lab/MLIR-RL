from llm_action.src.actions.v0.implementation.name import Name

from fastmcp import FastMCP

mcp = FastMCP("mlir-rl-action-space")

@mcp.tool()
def name_tool(code: str, parameters: dict) -> tuple[bool, str, bool]:
    """
    Applies <name> action on MLIR Code.
    
    Args:
        code (str): The MLIR code.
        parameters (dict): Parameters for the action, <adjust based on the action parameters definition>.
    
    Returns:
        tuple[bool, str, bool]: (precondition, transformed code, postcondition)
    """
    action = Name()
    if action.precondition(code, parameters):
            transformed_code = action.implement(code, parameters)        
            return True, transformed_code, action.postcondition(code, transformed_code, parameters)
    else:
        return False, code, False

if __name__ == "__main__":
    mcp.run()
