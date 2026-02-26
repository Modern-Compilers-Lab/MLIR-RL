import json
from pprint import pprint
from typing import Union

from agno.agent import Agent

from llm_action.src.config import CLAUDE_LLM_MODEL, GEMINI_LLM_MODEL
from llm_action.src.llm import get_claude_llm, get_gemini_llm
from llm_action.src.prompts.documentation_lookup import get_documentation_lookup_system_prompt
from llm_action.src.tools.transformation import lookup_transformation

from llm_action.src.utils.log import logger
from llm_action.src.models import ClaudeModel, GeminiModel
from llm_action.src.utils.parse import parse_action_implementation_output
from llm_action.src.utils.persistence import load_kernel_code_template, save_documentation_lookup_result
from llm_action.src.models import ActionPackage, ActionEnumeration

class DocumentationLookupAgent:
    def __init__(self, llm_model: Union[ClaudeModel, GeminiModel] = GEMINI_LLM_MODEL):
        self.name = "Documentation Lookup Agent"
        self.description = "An agent that looks up MLIR Transform dialect documentation for code transformations and optimizations."
        if isinstance(llm_model, ClaudeModel):
            self.model = get_claude_llm(llm_model=llm_model)
        elif isinstance(llm_model, GeminiModel):
            self.model = get_gemini_llm(llm_model=llm_model)
        else:
            raise ValueError(f"Unsupported LLM model: {llm_model}")
        self.agent = Agent(
            name=self.name,
            description=self.description,
            model=self.model,
            instructions=get_documentation_lookup_system_prompt(),
            tools=[lookup_transformation],
            add_history_to_context=False,
            num_history_runs=0,
            markdown=True,
        )
        
class DocumentationLookupAgentWrapper:
    def __init__(self, llm_model: Union[ClaudeModel, GeminiModel] = GEMINI_LLM_MODEL):
        self.documentation_lookup_agent = DocumentationLookupAgent(llm_model=llm_model)
        logger.info("[Agent] Documentation Lookup Agent initialized")
        
    def run(self, task: str) -> str:
        """
        Run the Documentation Lookup Agent for a code_template
        """

        response = self.documentation_lookup_agent.agent.run(
            input=task,
        )
        raw_content = response.content
        return raw_content

if __name__ == "__main__":
    llm_model = GeminiModel.GEMINI_2_5_FLASH
    agent_wrapper = DocumentationLookupAgentWrapper(llm_model=llm_model)
    task = "How to do vectorization in MLIR Transform dialect?"
    
    print(f"=== Running Documentation Lookup Agent using {llm_model.value} Model ===")
    response = agent_wrapper.run(task)
    print("=== Agent Response ===")
    print(response)
    save_path = save_documentation_lookup_result(task, response, model=llm_model)
    print(f"=== Response saved to: {save_path} ===")
