import json
from typing import Union

from agno.agent import Agent

from llm_action.src.actions.base import ActionBase
from llm_action.src.llm import get_claude_llm, get_gemini_llm, get_groq_llm
from llm_action.src.models import ClaudeModel, GeminiModel, GroqModel
from llm_action.src.prompts.parametrizer import get_parametrizer_system_prompt
from llm_action.src.utils.parse import extract_json_block


class ParametrizerAgent:
    def __init__(self, llm_model: Union[ClaudeModel, GeminiModel, GroqModel] = GroqModel.LLAMA_8b):
        if isinstance(llm_model, GeminiModel):
            model = get_gemini_llm(llm_model)
        elif isinstance(llm_model, GroqModel):
            model = get_groq_llm(llm_model)
        else:
            model = get_claude_llm(llm_model)

        self.agent = Agent(
            name="MLIR Action Parametrizer",
            description="Generates parameters for MLIR transformation actions",
            model=model,
            instructions=get_parametrizer_system_prompt(),
            markdown=False,
        )

    def parametrize(self, code: str, action_class: type[ActionBase], history: list[tuple[str, dict]]) -> dict:
        schema = action_class.parameters()
        if not schema:
            return {}

        prompt = self._build_prompt(code, action_class, history, schema)
        response = self.agent.run(input=prompt)
        text = response.content.strip()
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return json.loads(extract_json_block(text))
        return json.loads(text[start : end + 1])

    @staticmethod
    def _build_prompt(code: str, action_class: type[ActionBase], history: list[tuple[str, dict]], schema: dict) -> str:
        hist = "\n".join(f"  {i+1}. {n}({json.dumps(p)})" for i, (n, p) in enumerate(history)) or "  (none)"
        return (
            f"MLIR Code:\n```\n{code}\n```\n\n"
            f"Action: {action_class.__name__}\n"
            f"Description: {action_class.__doc__ or 'N/A'}\n\n"
            f"Parameter schema:\n{json.dumps(schema, indent=2)}\n\n"
            f"Previous actions:\n{hist}\n\n"
            f"Generate the parameters JSON."
        )
