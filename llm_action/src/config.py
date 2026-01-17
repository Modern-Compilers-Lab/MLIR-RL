from llm_action.src.models import ClaudeModel

# LLM
CLAUDE_LLM_MODEL = ClaudeModel.HAIKU
CLAUDE_LLM_TEMPERATURE = 1.0

# Execution
CODE_TRANSFORM_TIMEOUT = 10  # seconds
CODE_EXECUTION_TIMEOUT = 10  # seconds

# History
NUM_HISTORY_RUNS = 10

# Cache
USE_ACTION_ENUMERATION_CACHE = False

# Cached run paths
ACTION_ENUMERATION_CACHE = "llm_action/results/action_enumeration/mixed/claude-haiku-4-5/a0ea713d/action_enumeration.json"

# Verbose logging
TOOL_VERBOSE = True