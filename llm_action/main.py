from pprint import pprint

from llm_action.src.config import USE_ACTION_ENUMERATION_CACHE

from llm_action.src.utils.misc import random_id
from llm_action.src.utils.persistence import save_action_enumeration_result, save_action_implementation_result
from llm_action.src.agents.action_enumeration import ActionEnumerationAgentWrapper
from llm_action.src.agents.action_implementation import ActionImplementationAgentWrapper
from llm_action.src.prompts.representation import get_training_code_templates_representation

if __name__ == "__main__":
    runs_id = random_id()
    
    action_enumeration_agent_wrapper = ActionEnumerationAgentWrapper()
    
    print("=== Running Action Enumeration Agent ===")
    reasoning, action_enumeration = action_enumeration_agent_wrapper.run(get_training_code_templates_representation())
    print("===== Reasoning =====")
    print(reasoning)
    print("=== Action Enumeration Agent Response ===")
    pprint(action_enumeration.model_dump())
    if not USE_ACTION_ENUMERATION_CACHE:
        save_path = save_action_enumeration_result(reasoning, action_enumeration, run_id=runs_id)
        print(f"=== Response saved to: {save_path} ===")
        
    print("=== Running Action Implementation Agent ===")
    for intent in action_enumeration.intents:
        print(f"--- Implementing {intent.name} Actions ---")
        for transformation in intent.transformations:
            action_implementation_agent_wrapper = ActionImplementationAgentWrapper()
            reasoning, action_package, action_python_implementation = action_implementation_agent_wrapper.run(
                get_training_code_templates_representation(),
                intent,
                transformation
            )
            print(f"=== Implementation Agent Response for {transformation.name} Action ===")
            print("===== Reasoning =====")
            print(reasoning)
            print("===== Action Package =====")
            pprint(action_package.model_dump())
            print("===== Action Python =====")
            pprint(action_python_implementation)
            save_path = save_action_implementation_result(reasoning, action_package, action_python_implementation, run_id=runs_id, save_to_playground=True)
            print(f"=== Response saved to: {save_path} ===")
