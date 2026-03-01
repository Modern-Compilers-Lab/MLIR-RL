from llm_action.src.models import ClaudeModel, GeminiModel

# LLM
CLAUDE_LLM_MODEL = ClaudeModel.HAIKU
CLAUDE_LLM_TEMPERATURE = 1.0

GEMINI_LLM_MODEL = GeminiModel.GEMINI_2_5_FLASH
GEMINI_LLM_TEMPERATURE = 1.0

# Execution
N_CORES = 16
CODE_TRANSFORM_TIMEOUT = 10  # seconds
CODE_EXECUTION_TIMEOUT = 10  # seconds

# Transformation
VECTORIZATION_SIZE_LIMIT = 1024  # Max vectorization size to consider for transformations

# History
NUM_HISTORY_RUNS = 10

# Cache
USE_ACTION_ENUMERATION_CACHE = False

# Cached run paths
ACTION_ENUMERATION_CACHE = "llm_action/results/action_enumeration/mixed/claude-haiku-4-5/a0ea713d/action_enumeration.json"

# Verbose logging
TOOL_VERBOSE = True

# Web
MLIR_TRANSFORM_DIALECT_DOCS_URL = "https://mlir.llvm.org/docs/Dialects/Transform/"