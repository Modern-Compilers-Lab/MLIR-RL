import json
from pprint import pprint
from typing import AsyncGenerator, Optional, Tuple

from agno.agent import Agent, RunResponseEvent

from llm_action.src.config import CLAUDE_LLM_MODEL
from llm_action.src.llm import get_claude_llm
from llm_action.src.prompts.action_implementation import get_layer2_system_prompt
from llm_action.src.prompts.representation import get_optimization_intent_representation, get_transformation_representation, get_training_code_templates_representation

from llm_action.src.utils.log import logger
from llm_action.src.models import KernelType, ClaudeModel, OptimizationIntent, Transformation
from llm_action.src.utils.persistence import load_kernel_code_template, save_action_implementation_result, load_cached_action_enumeration
from llm_action.src.utils.parse import parse_action_implementation_output
from llm_action.src.models import ActionPackage, ActionEnumeration

from llm_action.src.tools import transform_code, execute_code, measure_speedup

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
            tools=[transform_code, execute_code, measure_speedup],
            memory=None,
            storage=None,
            add_history_to_messages=False,
            num_history_runs=0,
            show_tool_calls=True,
            markdown=True,
        )
        
        self.tool_execution = False

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
                message=f"Code Template: {code_template}\n{get_optimization_intent_representation(optimization_intent)}\n{get_transformation_representation(transformation)}",
            )
            raw_content = response.content
            reasoning, action_package, action_python_implementation = parse_action_implementation_output(raw_content)
            
        return reasoning, action_package, action_python_implementation

    async def run_stream(self, benchmark_code: str) -> AsyncGenerator[str, None]:
        """
        Run the Action Enumeration Agent for a benchmark_code
        """
        
        response_stream = await self.action_enumeration_agent.agent.arun(
            message=benchmark_code,
            stream=True,
            stream_intermediate_steps=True,
        )

        agent_response = ""
        
        async for event in response_stream:
            output = self._format_event(event)
            if output:
                if output.get('type') == 'content':
                    agent_response += output['content']
                yield output
        
        # logger.info(f"[Agent]: Response: {agent_response}")

    def _format_event(self, event: RunResponseEvent) -> Optional[str]:
        """
        Converts an agent event into a string for streaming/yielding.
        """
        match event.event:
            
            # Run Events
            case "RunStarted":
                return {
                    "type": "run",
                    "status": "started"
                }
            case "RunResponseContent":
                if not self.mlir_llm_agent.tool_execution:
                    return {
                        "type": "content",
                        "content": event.content
                    }
            case "RunCompleted":
                return {
                    "type": "run",
                    "status": "completed"
                }
            case "RunError":
                return {
                    "type": "error",
                    "error": f"{event.content}"
                }
            case "RunCanceled":
                return {
                    "type": "run",
                    "status": f"canceled"
                }
            case "RunPaused":
                return {
                    "type": "run",
                    "status": "paused"
                }
            case "RunContinued":
                return {
                    "type": "run",
                    "status": "continued"
                }

            # Tool Events
            case "ToolCallStarted":

                self.mlir_llm_agent.tool_execution = True
                return {
                    "type": "tool",
                    "status": "started",
                    "name": event.tool.tool_name,
                    "arguments": event.tool.tool_args,
                }
            case "ToolCallCompleted":
                self.mlir_llm_agent.tool_execution = False

                try:
                    tool_result = json.loads(event.tool.result)
                except json.JSONDecodeError:
                    tool_result = event.tool.result
                    
                return {
                    "type": "tool",
                    "status": "completed",
                    "name": event.tool.tool_name,
                    "result": tool_result,
                }
                
            # Reasoning Events
            case "ReasoningStarted":
                return {
                    "type": "reasoning",
                    "status": "started",
                }
            case "ReasoningStep":
                return {
                    "type": "reasoning",
                    "status": "step",
                    "content": event.content
                }
            case "ReasoningCompleted":
                return {
                    "type": "reasoning",
                    "status": "completed",
                    "content": event.content
                }

            # Memory Events
            case "MemoryUpdateStarted":
                return {
                    "type": "memory_update",
                    "status": "started",
                }
            case "MemoryUpdateCompleted":
                return {
                    "type": "memory_update",
                    "status": "completed",
                    "content": event.content
                }
            
            # Default case
            case _:
                return f"Unhandled event: {event.event}"

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
    save_path, candidate_path = save_action_implementation_result(reasoning, action_package, action_python_implementation, KernelType.MIXED, llm_model, save_to_playground=True)
    print(f"=== Response saved to: {save_path} ===")
    if candidate_path:
        print(f"=== Candidate action also saved to: {candidate_path} ===")
