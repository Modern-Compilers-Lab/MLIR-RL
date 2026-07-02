# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

# Import modules
import os
import logging
import signal
import torch
from typing import Optional

def _sigabrt_handler(signum, frame):
    raise RuntimeError("MLIR native code crashed (SIGABRT) — caught and continuing")
signal.signal(signal.SIGABRT, _sigabrt_handler)
from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger
from utils.config import Config
from utils.log import print_info, print_success
from utils.implementation import get_agent_runs_root, get_autoschedular_impl, import_autoschedular_module
from datetime import timedelta
from time import time
import json
import shutil

AUTOSCHEDULER_IMPL = get_autoschedular_impl()
Execution = import_autoschedular_module("execution", AUTOSCHEDULER_IMPL).Execution
Model = import_autoschedular_module("model", AUTOSCHEDULER_IMPL).HiearchyModel
device = import_autoschedular_module("", AUTOSCHEDULER_IMPL).device
evaluate_benchmarks = import_autoschedular_module("ppo", AUTOSCHEDULER_IMPL).evaluate_benchmarks
Benchmarks = import_autoschedular_module("benchmarks", AUTOSCHEDULER_IMPL).Benchmarks

logging.basicConfig(
    filename=f"logs/{os.getenv('SLURM_JOB_NAME', 'interactive')}_{os.environ['SLURM_JOB_ID']}.debug",
    filemode="w",
    format="${asctime} - [${levelname}]    ${name}: ${message}",
    datefmt="%m-%d %H:%M",
    style='$',
    level=logging.DEBUG
)

# Initialize singleton classes
cfg = Config()
fl = FileLogger()
dm = DaskManager()


# Data loading
def load_eval_data():
    return Benchmarks(is_training=False)


def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
    return None


eval_data = dm.run_and_register_to_workers(load_eval_data)
print_info(f"Loaded {len(eval_data)} benchmark(s): {', '.join(d.bench_name for d in eval_data.data)}")
main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

# Initialize execution singleton
Execution(fl.exec_data_file, main_exec_data)

# Setup signal handler again (MLIR imports / initialization overrides Python's signal handler)
signal.signal(signal.SIGABRT, _sigabrt_handler)

# Prepare logging
print_info(f"Config: {cfg}")
print_info(f"Autoscheduler implementation: {AUTOSCHEDULER_IMPL}")
print_success(f'Logging to: {fl.run_dir}')

# Setup torch
torch.set_grad_enabled(False)
torch.set_num_threads(4)

# Initiate model
model = Model().to(device)
print_success("Model initialized")

# Start evaluation
eval_dir = os.getenv('EVAL_DIR')
if eval_dir is None:
    # Derive from config: find the latest run_N for the selected implementation
    # whose models/ dir has .pt files.
    # (the current eval run's models/ dir is always empty)
    _agent_root = str(get_agent_runs_root(cfg.results_dir, AUTOSCHEDULER_IMPL))
    if not os.path.isdir(_agent_root):
        raise ValueError(
            "No implementation run directory found. "
            f"Expected: {_agent_root}. Run training first or set EVAL_DIR explicitly."
        )
    _runs = sorted(
        [d for d in os.listdir(_agent_root) if d.startswith('run_') and d.split('_')[-1].isdigit()],
        key=lambda x: int(x.split('_')[1])
    )
    _candidates = [
        os.path.join(_agent_root, d, 'models') for d in _runs
        if any(f.endswith('.pt') for f in os.listdir(os.path.join(_agent_root, d, 'models'))
               if os.path.isdir(os.path.join(_agent_root, d, 'models')))
    ]
    if not _candidates:
        raise ValueError(
            "No run with saved model checkpoints found in results_dir. "
            "Set EVAL_DIR explicitly or run training first. "
            f"Looked under: {_agent_root}"
        )
    eval_dir = _candidates[-1]
    print(f"EVAL_DIR not set; using: {eval_dir}")
eval_dir = os.path.abspath(eval_dir)

# Read the files in the evaluation directory
eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.pt')]

# Order files
eval_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

# Filter and sort checkpoints
import re
EVAL_LAST_ONLY = os.getenv("EVAL_LAST_ONLY", "").strip().lower() in ("1", "true", "yes")
EVAL_STRIDE = int(os.getenv("EVAL_STRIDE", "100"))
EVAL_START = int(os.getenv("EVAL_START", "0"))
EVAL_END = int(os.getenv("EVAL_END", "999999"))

def get_model_step(f):
    match = re.search(r'model_(\d+)\.pt', f)
    return int(match.group(1)) if match else -1

# 1. Get all valid models with their steps
all_models = []
for f in os.listdir(eval_dir):
    if f.endswith('.pt'):
        step = get_model_step(f)
        if step >= 0:
            all_models.append((f, step))

# 2. Sort by step
all_models.sort(key=lambda x: x[1])

# 3. Apply range and stride
filtered = [
    f for f, step in all_models 
    if EVAL_START <= step <= EVAL_END and step % EVAL_STRIDE == 0
]

# 4. Always include the absolute last model if within range
if all_models:
    last_f, last_step = all_models[-1]
    if EVAL_START <= last_step <= EVAL_END and last_f not in filtered:
        filtered.append(last_f)
    # Re-sort after potentially adding last
    filtered.sort(key=lambda f: get_model_step(f))

eval_files = filtered
if EVAL_LAST_ONLY:
    eval_files = [eval_files[-1]] if eval_files else []
    if eval_files:
        print_info(f"EVAL_LAST_ONLY: evaluating only {eval_files[0]}")
else:
    print_info(f"Checkpoints to evaluate: {len(eval_files)} (stride={EVAL_STRIDE}, range=[{EVAL_START}, {EVAL_END}])")

# Resumption: track completed checkpoints in a state file
EVAL_FORCE = os.getenv("EVAL_FORCE", "").strip().lower() in ("1", "true", "yes")
eval_logs_dir = os.path.join(fl.logs_dir, 'eval')
os.makedirs(eval_logs_dir, exist_ok=True)
completed_file = os.path.join(eval_logs_dir, '_eval_checkpoint.txt')
completed = set()
if EVAL_FORCE:
    if os.path.exists(completed_file):
        os.remove(completed_file)
        print_info("EVAL_FORCE: deleted _eval_checkpoint.txt, re-evaluating all")
    else:
        print_info("EVAL_FORCE: re-evaluating all checkpoints")
elif os.path.exists(completed_file):
    with open(completed_file) as f:
        completed = set(line.strip() for line in f if line.strip())
    already = [f for f in eval_files if f in completed]
    if already:
        print_info(f"Resuming: {len(already)} checkpoints already evaluated, skipping")

pending_files = [f for f in eval_files if f not in completed]

iter_time_dlt = 0
elapsed_dlt = 0
eta_dlt = 0
overall_start = time()
models_count = len(pending_files)
for step, model_file in enumerate(pending_files):
    print_info(
        f"- Evaluation {step + 1}/{models_count}"
        f" ({100 * (step + 1) / models_count:.2f}%)"
        f" ({iter_time_dlt}/it) ({elapsed_dlt} < {eta_dlt})",
        flush=True
    )

    main_start = time()

    model_path = os.path.join(eval_dir, model_file)
    if not os.path.exists(model_path):
        print_info(f"Model file {model_path} does not exist. Skipping.")
        continue
    checkpoint = torch.load(model_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    evaluate_benchmarks(model, eval_data)

    main_end = time()
    iter_time = main_end - main_start
    elapsed = main_end - overall_start
    eta = elapsed * (models_count - step - 1) / (step + 1)
    iter_time_dlt = timedelta(seconds=iter_time)
    elapsed_dlt = timedelta(seconds=int(elapsed))
    eta_dlt = timedelta(seconds=int(eta))

    # Mark checkpoint as completed for resumption
    with open(completed_file, 'a') as f:
        f.write(model_file + '\n')

# Post-process: if --checkpoint mode, save results to agent_dir/eval/checkpoint_<N>.json
_ckpt = os.getenv("EVAL_CHECKPOINT")
if _ckpt:
    _label = os.getenv("EVAL_LABEL")
    _suffix = f"_{_label}" if _label else ""
    _agent_dir = os.path.dirname(eval_dir)  # eval_dir = agent/models/ → _agent_dir = agent/
    eval_root = os.path.join(_agent_dir, "eval")
    ckpt_file = os.path.join(eval_root, f"checkpoint_{_ckpt}{_suffix}.json")
    if os.path.exists(ckpt_file) and not EVAL_FORCE:
        print_info(f"checkpoint_{_ckpt}{_suffix}.json already exists, skipping post-process")
    else:
        if os.path.exists(ckpt_file) and EVAL_FORCE:
            print_info(f"EVAL_FORCE: overwriting {ckpt_file}")
        os.makedirs(eval_root, exist_ok=True)
        _eval_file = f"eval_exec_times_{_ckpt}.json"
        src_eval = os.path.join(fl.logs_dir, "eval", _eval_file)
        if os.path.exists(src_eval):
            shutil.copy2(src_eval, ckpt_file)
            print_success(f"Saved eval results to eval/checkpoint_{_ckpt}{_suffix}.json")

        # Copy key log files
        src_logs = os.path.join(fl.logs_dir, "eval")
        if os.path.isdir(src_logs):
            dst_logs = os.path.join(eval_root, "logs", f"{_ckpt}{_suffix}")
            os.makedirs(os.path.dirname(dst_logs), exist_ok=True)
            if os.path.exists(dst_logs):
                shutil.rmtree(dst_logs)
            os.makedirs(dst_logs)
            for log_file in ("final_speedup", "average_speedup", "arithmetic_mean_speedup"):
                src = os.path.join(src_logs, log_file)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(dst_logs, log_file))
            print_info(f"Copied logs to eval/logs/{_ckpt}{_suffix}/")

    print_info("Evaluation done.")
