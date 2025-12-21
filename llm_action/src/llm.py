from agno.models.anthropic import Claude

from llm_action.src.keys import ANTHROPIC_API_KEY
from llm_action.src.config import CLAUDE_LLM_MODEL, CLAUDE_LLM_TEMPERATURE

def get_claude_llm():
    llm = Claude(
        id=CLAUDE_LLM_MODEL,
        temperature=CLAUDE_LLM_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
    )
    return llm
