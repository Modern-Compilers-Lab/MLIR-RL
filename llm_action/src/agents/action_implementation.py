import json
from pprint import pprint
from typing import AsyncGenerator, Optional, Tuple

from agno.agent import Agent

from llm_action.src.config import CLAUDE_LLM_MODEL
from llm_action.src.llm import get_claude_llm
from llm_action.src.prompts.action_implementation import get_layer2_system_prompt
from llm_action.src.prompts.representation import get_optimization_intent_representation, get_transformation_representation, get_training_code_templates_representation

from llm_action.src.utils.log import logger
from llm_action.src.models import KernelType, ClaudeModel, OptimizationIntent, Transformation
from llm_action.src.utils.persistence import load_kernel_code_template, save_action_implementation_result, load_cached_action_enumeration
from llm_action.src.utils.parse import parse_action_implementation_output
from llm_action.src.models import ActionPackage, ActionEnumeration

from llm_action.src.tools.transformation import transform_code, execute_code, measure_speedup
from llm_action.src.tools.agent_as_tool import delegate_documentation_lookup

class ActionImplementationAgent:
    def __init__(self, llm_model: ClaudeModel = CLAUDE_LLM_MODEL):
        self.name = "Layer 2 - Action Implementation Agent"
        self.description = "An agent that implements atomic transformations as executable actions for MLIR code templates."
        self.model = get_claude_llm(llm_model=llm_model)
        self.agent = Agent(
            name=self.name,
            description=self.description,
            model=self.model,
            instructions=get_layer2_system_prompt(),
            tools=[delegate_documentation_lookup, transform_code, execute_code, measure_speedup],
            add_history_to_context=False,
            num_history_runs=0,
            markdown=True,
        )
        
class ActionImplementationAgentWrapper:
    def __init__(self, llm_model: ClaudeModel = CLAUDE_LLM_MODEL, use_cache: bool = False):
        self.action_implementation_agent = ActionImplementationAgent(llm_model=llm_model)
        self.use_cache = use_cache
        logger.info("[Agent] Action Implementation Agent initialized")
        
    def run(self, code_template: str, optimization_intent: OptimizationIntent, transformation: Transformation) -> Tuple[str, ActionPackage, str]:
        """
        Run the Action Implementation Agent for a code_template
        """
        if self.use_cache:
            pass
        else:
            response = self.action_implementation_agent.agent.run(
                input=f"Code Template: {code_template}\n{get_optimization_intent_representation(optimization_intent)}\n{get_transformation_representation(transformation)}",
            )
            raw_content = response.content
            reasoning, action_package, action_python_implementation = parse_action_implementation_output(raw_content)
            
        return reasoning, action_package, action_python_implementation

if __name__ == "__main__":
    llm_model = ClaudeModel.HAIKU
    agent_wrapper = ActionImplementationAgentWrapper(llm_model=llm_model)
    # code_template = load_kernel_code_template(KernelType.CONV2D)
    cached_action_enumeration: ActionEnumeration = load_cached_action_enumeration()
    optimization_intent = cached_action_enumeration.intents[0]
    transformation = optimization_intent.transformations[0]
    
    print(f"=== Running Action Implementation Agent using {llm_model.value} Model ===")
    reasoning, action_package, action_python_implementation = agent_wrapper.run(
        get_training_code_templates_representation(),
        optimization_intent,
        transformation
    )
    print("=== Agent Response ===")
    print("===== Action Package =====")
    pprint(action_package.model_dump())
    print("===== Action Python =====")
    pprint(action_python_implementation)
    save_path, candidate_path = save_action_implementation_result(reasoning, action_package, action_python_implementation, KernelType.MIXED, llm_model, save_to_playground=False)
    print(f"=== Response saved to: {save_path} ===")
    if candidate_path:
        print(f"=== Candidate action also saved to: {candidate_path} ===")
