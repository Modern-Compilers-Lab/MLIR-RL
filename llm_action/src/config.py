from pathlib import Path
from llm_action.src.models import ClaudeModel, GeminiModel, GroqModel

# System
CONDA_ENV = "mlir"

# LLM
CLAUDE_LLM_MODEL = ClaudeModel.HAIKU
CLAUDE_LLM_TEMPERATURE = 1.0

GEMINI_LLM_MODEL = GeminiModel.GEMINI_2_5_FLASH
GEMINI_LLM_TEMPERATURE = 1.0

GROQ_LLM_MODEL = GroqModel.GPT_OSS_120B
GROQ_LLM_TEMPERATURE = 1.0

# Execution
N_CORES = 28
CODE_TRANSFORM_TIMEOUT = 10  # seconds
CODE_EXECUTION_TIMEOUT = 10  # seconds
DASK_WAIT_TIMEOUT = 300  # seconds to wait for Dask cluster to be ready
SLURM_TIMEOUT = 300  # seconds to wait for SLURM job to complete

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

# Paths
PROJECT_ROOT = Path("/scratch/kb5213/workspace/MLIR-RL/")

DATA_DIR = PROJECT_ROOT / "llm_action" / "data" / "benchmarks"

RL_RESULTS_DIR = PROJECT_ROOT / "llm_action" / "results" / "rl"
MLIR_SLURM_LOG_DIR = PROJECT_ROOT / "llm_action" / "logs" / "jobs" / "mlir"
TORCH_SLURM_LOG_DIR = PROJECT_ROOT / "llm_action" / "logs" / "jobs" / "torch"

DASK_TMP_DIR = PROJECT_ROOT / "llm_action" / "tmp" / "dask"
MLIR_TMP_DIR = PROJECT_ROOT / "llm_action" / "tmp"

TORCH_SCRIPT = PROJECT_ROOT / "llm_action" / "scripts" / "torch.sh"
MLIR_SCRIPT = PROJECT_ROOT / "llm_action" / "scripts" / "mlir.sh"

# State
## Max number of loops
L = 3
## Max number of load/store operations
LS = 3 
## Max number of dimensions for load/store operations
LSD = 3 
## Max number of transformation steps in an episode
MAX_STEPS = 7 
## Arithmetic operations to track in the observation
ARITH_OPS = ["+", "-", "*", "/", "exp"] 
## Number of operation types tracked in the observation (see OperationType enum in state_extractor)
NUM_OP_TYPES = 2  # Generic, Matmul
## Size of the operation features in the observation vector
OP_FEATURES_SIZE = NUM_OP_TYPES + L + L + LS * LSD * L + LS * LSD * L + len(ARITH_OPS)

# Action
MAX_PARAM_SLOTS = 3  # Upper bound on parameter slots per action
MAX_VOCAB_SIZE_PER_SLOT = 5  # Upper bound on vocabulary size per parameter slot