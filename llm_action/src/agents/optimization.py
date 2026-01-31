import json
from pprint import pprint
from typing import AsyncGenerator, Optional, Tuple

from agno.agent import Agent

from llm_action.src.config import CLAUDE_LLM_MODEL
from llm_action.src.llm import get_claude_llm
from llm_action.src.prompts.optimization import get_optimization_system_prompt
from llm_action.src.prompts.representation import get_code_representation

from llm_action.src.utils.log import logger
from llm_action.src.models import KernelType, ClaudeModel
from llm_action.src.utils.persistence import load_kernel_code

from llm_action.src.tools.transformation import transform_code, execute_code, measure_speedup
from llm_action.src.tools.agent_as_tool import delegate_documentation_lookup

class OptimizationAgent:
    def __init__(self, llm_model: ClaudeModel = CLAUDE_LLM_MODEL):
        self.name = "MLIR Optimization Agent"
        self.description = "An agent that optimizes MLIR code."
        self.model = get_claude_llm(llm_model=llm_model)
        self.agent = Agent(
            name=self.name,
            description=self.description,
            model=self.model,
            instructions=get_optimization_system_prompt(),
            tools=[delegate_documentation_lookup, transform_code, execute_code, measure_speedup],
            add_history_to_context=False,
            num_history_runs=0,
            max_tool_calls_from_history=0,
            markdown=True,
        )
        
class OptimizationAgentWrapper:
    def __init__(self, llm_model: ClaudeModel = CLAUDE_LLM_MODEL, use_cache: bool = False):
        self.optimization_agent = OptimizationAgent(llm_model=llm_model)
        logger.info("[Agent] Optimization Agent initialized")
        
    def run(self, code: str) -> str:
        """
        Run the Optimization Agent for a given MLIR code snippet.
        """
        response = self.optimization_agent.agent.run(
            input=code,
        )
        raw_content = response.content
            
        return raw_content

if __name__ == "__main__":
    llm_model = ClaudeModel.HAIKU
    agent_wrapper = OptimizationAgentWrapper(llm_model=llm_model)
    code = load_kernel_code(KernelType.MATMUL)
    
    print(f"=== Running Optimization Agent using {llm_model.value} Model ===")
    raw_content = agent_wrapper.run(code + "Please test out simple tiling and vectorization optimizations, this is just a unit test.")
    print("=== Agent Response ===")
    print(raw_content)
    
    # save_path, candidate_path = save_action_implementation_result(reasoning, action_package, action_python_implementation, KernelType.MIXED, llm_model, save_to_playground=False)
    # print(f"=== Response saved to: {save_path} ===")
    # if candidate_path:
    #     print(f"=== Candidate action also saved to: {candidate_path} ===")
