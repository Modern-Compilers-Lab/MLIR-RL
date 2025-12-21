import asyncio

from llm_action.src.agent import AgentWrapper

if __name__ == "__main__":
    agent = AgentWrapper()
    agent.run_playground()
