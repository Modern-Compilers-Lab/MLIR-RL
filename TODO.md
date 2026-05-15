You context is limited to the package: llm_action/

First understand clearly the automatic action generation pipeline with its 3 layers, trace from the scripts in llm_action/scripts/:
- action enumeration
- action implementation
- schedule exploration

The actions that are generated follow the template in llm_action/src/actions/v0

Then the action space integrates seamlessly with the RL environment: llm_action/src/env/

We want to work on RL training robustness, especially regarding parameters masking, to elaborate with an example, when the agent tiles with non-divisibles tiles -> produces dynamic vectors when trying to vectorize which fails on MLIR. We want to solve such cases from first principles and on the action generation pipeline itself.

1- Try to understand all masking logics present in the current codebase and provide a detailed overview how masking operates on the policy network level
2- Understand clearly the prompting style and guidelines we make in order to respect how we instruct the different level agents
3- Most importantly, suggest an approach on the action generation level, in order to embed in the Action base class (as an attribute/method) on how to solve such issues by making the action automatically set a mask that robustly ensures/reduces compilation bugs and improve sampling efficieny and improves RL training. and how to integrate this with the enviornment such as communicating the loops and their bounds.