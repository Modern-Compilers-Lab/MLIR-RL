import json
from typing import AsyncGenerator, Optional

from agno.agent import Agent, RunResponseEvent
from agno.playground import Playground, PlaygroundSettings

from llm_action.src.llm import get_claude_llm
from llm_action.src.tools import transform_code, execute_code, measure_speedup
from llm_action.src.prompt import SYSTEM_INSTRUCTIONS
from llm_action.src.config import NUM_HISTORY_RUNS

from llm_action.src.utils.log import logger

class MLIR_LLM_Agent:
    def __init__(self):
        self.name = "MLIR LLM Agent"
        self.description = "An agent that assists with MLIR automatic code optimization."
        self.model = get_claude_llm()
        self.agent = Agent(
            name=self.name,
            description=self.description,
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            tools=[measure_speedup, transform_code, execute_code],
            memory=None,
            storage=None,
            add_history_to_messages=True,
            num_history_runs=NUM_HISTORY_RUNS,
            show_tool_calls=True,
            markdown=True,
        )
        
        self.tool_execution = False

class AgentWrapper:
    def __init__(self):
        self.mlir_llm_agent = MLIR_LLM_Agent()
        # logger.info("[Agent] MLIR LLM Agent initialized")

    async def run_stream(self, benchmark_code: str) -> AsyncGenerator[str, None]:
        """
        Run the Agno agents for a benchmark_code
        """
        
        response_stream = await self.mlir_llm_agent.agent.arun(
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

    def run_playground(self) -> None:
        """
        Run the agent playground server.
        """
        
        playground = Playground(
            agents=[self.mlir_llm_agent.agent],
            settings=PlaygroundSettings(env="dev")
        )
        app = playground.get_app()

        playground.serve(app)
