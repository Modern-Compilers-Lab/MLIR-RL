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

# ---- Timeouts (all in seconds) ----

## MLIR Python bindings (in-process / spawn-child via BindingsProcess)
CODE_TRANSFORM_TIMEOUT = 60                          # one transform-dialect application
CODE_EXECUTION_TIMEOUT = 60                          # one bufferized lower+execute
CODE_BUFFERIZE_AND_EXECUTE_TIMEOUT = (               # combined budget for the Dask-worker round-trip
    CODE_TRANSFORM_TIMEOUT + CODE_EXECUTION_TIMEOUT
)

## SLURM (driver-side; NOT the .sh #SBATCH -t walltime)
SLURM_TIMEOUT = 300                                  # max wall time the driver waits for a SLURM job to leave squeue
SLURM_POLL_INTERVAL = 2                              # squeue poll cadence

## Dask
DASK_WAIT_TIMEOUT = 300                              # client.wait_for_workers at cluster startup
DASK_TIMEOUT = CODE_BUFFERIZE_AND_EXECUTE_TIMEOUT + 10  # outer Future.result must be >= inner BindingsProcess budget; +10s IPC slack

## Other
AST_DUMPER_TIMEOUT = 40                              # state_extractor.py subprocess to AST_DUMPER_BIN_PATH
HTTP_FETCH_TIMEOUT = 60                              # documentation scraping (requests.get)

# Transformation
VECTORIZATION_SIZE_LIMIT = 2048  # Max vectorization size to consider for transformations

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

# Log
## Max key width (chars) for the SB3 stdout logger table.
SB3_STDOUT_KEY_MAX_LENGTH = 60

# State
## Max number of loops the observation/action mask can address.
## Sized for the largest op family in the dataset: matmul=3, add=4, relu-generic=2/4, pooling_nchw_max=6, conv_2d_nchw_fchw=7.
L = 7
## Max rows per access table (loads or stores). Required >= max(n_loads, n_stores)
## across all op families: matmul/conv/add/pool=2 loads + 1 store, relu=1+1.
## So 2 is the actual minimum; 3 leaves one zero row of headroom.
LS = 2
## Max affine-map output dims per access (i.e. rank of the indexed tensor).
## Required: matmul=2, conv/add/pool/4D-relu=4. With LSD<4, _encode_access
## silently truncates to terms[:LSD] and drops the 4th dim's affine structure
## (W in NCHW), so 4 is the minimum for mixed-op correctness.
LSD = 4
## Max number of transformation steps in an episode
MAX_STEPS = 7
## Arithmetic operations to track in the observation
ARITH_OPS = ["+", "-", "*", "/", "exp"]
## Number of operation types tracked in the observation. the enum currently has Generic, Matmul, Conv, Pooling, Add, Relu
NUM_OP_TYPES = 6
## Size of the operation features in the observation vector
OP_FEATURES_SIZE = NUM_OP_TYPES + L + L + LS * LSD * L + LS * LSD * L + len(ARITH_OPS)

# Action
## Upper bound on per-loop parameter slots in actions, min(n_loops, MAX_PARAM_SLOTS), so this must be >= L for the policy to address every loop in the largest op family.
MAX_PARAM_SLOTS = L
## Upper bound on vocabulary size per parameter slot
MAX_VOCAB_SIZE_PER_SLOT = 5