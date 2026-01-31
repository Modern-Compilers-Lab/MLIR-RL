import json
from pprint import pprint
from typing import AsyncGenerator, Optional, Tuple

from agno.agent import Agent

from llm_action.src.config import CLAUDE_LLM_MODEL, USE_ACTION_ENUMERATION_CACHE
from llm_action.src.llm import get_claude_llm
from llm_action.src.prompts.action_enumeration import get_layer1_system_prompt
from llm_action.src.prompts.representation import get_training_code_templates_representation

from llm_action.src.utils.log import logger
from llm_action.src.models import KernelType, ClaudeModel
from llm_action.src.utils.persistence import load_kernel_code_template, save_action_enumeration_result, load_cached_action_enumeration
from llm_action.src.utils.parse import parse_action_enumeration_output
from llm_action.src.models import ActionEnumeration

class ActionEnumerationAgent:
    def __init__(self, llm_model: ClaudeModel = CLAUDE_LLM_MODEL):
        self.name = "Layer 1 - Action Enumeration Agent"
        self.description = "An agent that enumerates optimization intents and atomic transformations for MLIR code templates."
        self.model = get_claude_llm(llm_model=llm_model)
        self.agent = Agent(
            name=self.name,
            description=self.description,
            model=self.model,
            instructions=get_layer1_system_prompt(),
            tools=[],
            add_history_to_context=False,
            num_history_runs=0,
            markdown=True,
        )
        
class ActionEnumerationAgentWrapper:
    def __init__(self, llm_model: ClaudeModel = CLAUDE_LLM_MODEL, use_cache: bool = USE_ACTION_ENUMERATION_CACHE):
        self.action_enumeration_agent = ActionEnumerationAgent(llm_model=llm_model)
        self.use_cache = use_cache
        logger.info("[Agent] Action Enumeration Agent initialized")
        
    def run(self, code_template: str) -> Tuple[str, ActionEnumeration]:
        """
        Run the Action Enumeration Agent for a code_template
        """
        if self.use_cache:
            reasoning = ""
            action_enumeration = load_cached_action_enumeration()
        else:
            response = self.action_enumeration_agent.agent.run(
                input=code_template,
            )
            raw_content = response.content
            reasoning, action_enumeration = parse_action_enumeration_output(raw_content)
            
        return reasoning, action_enumeration

if __name__ == "__main__":
    llm_model = ClaudeModel.HAIKU
    agent_wrapper = ActionEnumerationAgentWrapper(llm_model=llm_model)
    print(f"=== Running Action Enumeration Agent using {llm_model.value} Model ===")
    # code_template = load_kernel_code_template(KernelType.CONV2D)
    # response = agent_wrapper.run(code_template)
    reasoning, action_enumeration = agent_wrapper.run(get_training_code_templates_representation(include_instances=False))
    print("=== Agent Response Reasoning ===")
    pprint(reasoning)
    print("=== Agent Response Action Enumeration ===")
    pprint(action_enumeration.model_dump())
    save_path = save_action_enumeration_result(reasoning, action_enumeration, KernelType.MIXED, llm_model)
    print(f"=== Response saved to: {save_path} ===")
