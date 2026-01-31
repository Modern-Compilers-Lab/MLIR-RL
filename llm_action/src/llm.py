from agno.models.anthropic import Claude

from llm_action.src.keys import ANTHROPIC_API_KEY
from llm_action.src.config import CLAUDE_LLM_MODEL, CLAUDE_LLM_TEMPERATURE
from llm_action.src.models import ClaudeModel

def get_claude_llm(llm_model: ClaudeModel = CLAUDE_LLM_MODEL) -> Claude:
    llm = Claude(
        id=llm_model.value,
        temperature=CLAUDE_LLM_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
        betas=["context-management-2025-06-27"],
        context_management={
            "edits": [{
                "type": "clear_tool_uses_20250919",
                "trigger": {"type": "input_tokens", "value": 100000},
                "keep": {"type": "tool_uses", "value": 0},
                "exclude_tools": ["transform_code", "execute_code", "measure_speedup"],
                "clear_tool_inputs": True,
            }]
        },
    )
    return llm
