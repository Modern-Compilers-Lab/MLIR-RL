You context is limited to the package: llm_action/

First understand clearly the automatic action generation pipeline with its 3 layers, trace from the scripts in llm_action/scripts/:
- action enumeration
- action implementation
- schedule exploration

The actions that are generated follow the template in llm_action/src/actions/v0

Then the action space integrates seamlessly with the RL environment: llm_action/src/env/

<task>