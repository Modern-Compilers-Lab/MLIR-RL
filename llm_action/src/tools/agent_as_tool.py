from llm_action.src.utils.log import logger
from llm_action.src.config import TOOL_VERBOSE

from agno.tools import tool

from llm_action.src.models import ClaudeModel
from llm_action.src.agents.documentation_lookup import DocumentationLookupAgentWrapper 

@tool(
    name="delegate_documentation_lookup",
    description="""
    Delegates a documentation lookup task to the Documentation Lookup Agent.
    
    This tool forwards a specific documentation retrieval task to the Documentation Lookup Agent, which specializes in finding authoritative references for MLIR Transform dialect operations.
    
    Args:
        task: The documentation lookup task or question to be answered
        
    Returns:
        The response from the Documentation Lookup Agent containing the requested documentation information
    """,
    show_result=True,
    stop_after_tool_call=False
)
def delegate_documentation_lookup(task: str) -> str:
    agent = DocumentationLookupAgentWrapper(llm_model=ClaudeModel.HAIKU)
    if TOOL_VERBOSE:
        logger.info("[TOOL] Executing `delegate_documentation_lookup`")
        logger.info(f"[TOOL PARAM] Task: {task}")
    result = agent.run(task)
    if TOOL_VERBOSE:
        logger.info(f"[TOOL RESULT] Result: {result}")
    return result
