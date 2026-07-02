# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

# Import modules
import os
import sys

# Robust PATH restoration
os.environ["PATH"] = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:" + os.environ.get("PATH", "")

import logging
import signal
import torch
import re
from typing import Optional
from datetime import timedelta
from time import time

def _sigabrt_handler(signum, frame):
    raise RuntimeError("MLIR native code crashed (SIGABRT) — caught and continuing")
signal.signal(signal.SIGABRT, _sigabrt_handler)

from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger
from utils.config import Config
from utils.log import print_info, print_success
from utils.implementation import get_agent_runs_root, get_autoschedular_impl, import_autoschedular_module

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

def load_eval_data():
    return Benchmarks(is_training=False)

def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
    return None

eval_data = dm.run_and_register_to_workers(load_eval_data)
main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

Execution(fl.exec_data_file, main_exec_data)

# Setup signal handler again (MLIR imports / initialization overrides Python's signal handler)
signal.signal(signal.SIGABRT, _sigabrt_handler)

print_info(f"Config: {cfg}")
print_info(f"Autoscheduler implementation: {AUTOSCHEDULER_IMPL}")
print_success(f'Logging to: {fl.run_dir}')

torch.set_grad_enabled(False)
torch.set_num_threads(4)

model = Model().to(device)
print_success("Model initialized")

# --- ABLATION LOGIC START ---
eval_dir = os.getenv('EVAL_DIR', '').strip()
if not eval_dir:
     # Fallback only if EVAL_DIR is truly missing
    _agent_root = str(get_agent_runs_root(cfg.results_dir, AUTOSCHEDULER_IMPL))
    _runs = sorted([d for d in os.listdir(_agent_root) if d.startswith('run_') and d.split('_')[-1].isdigit()], key=lambda x: int(x.split('_')[1]))
    _candidates = [os.path.join(_agent_root, d, 'models') for d in _runs if os.path.isdir(os.path.join(_agent_root, d, 'models'))]
    eval_dir = _candidates[-1]

print_info(f"DEBUG: Current PATH: {os.environ.get('PATH')}")
print_info(f"DEBUG: Full ENV: {os.environ}")
eval_dir = os.path.abspath(eval_dir)
if not os.path.isdir(eval_dir):
    print(f"DEBUG: Directory check failed for: '{eval_dir}'")
    import subprocess
    print(f"DEBUG: ls parent: {subprocess.getoutput(f'ls -F /scratch/mb10856/MLIR-RL/results/experiment3/ 2>&1')}")
    print(f"DEBUG: ls agent: {subprocess.getoutput(f'ls -F /scratch/mb10856/MLIR-RL/results/experiment3/rl_autoschedular_v4_5_agent/ 2>&1')}")
    raise ValueError(f"EVAL_DIR is not a directory: '{eval_dir}'")

print_info(f"Evaluating models from: {eval_dir}")

# Filter and sort checkpoints
EVAL_LAST_ONLY = os.getenv("EVAL_LAST_ONLY", "false").strip().lower() in ("1", "true", "yes")
EVAL_STRIDE = 100
EVAL_START = int(os.getenv("EVAL_START", "0"))
EVAL_END = int(os.getenv("EVAL_END", "999999"))
EVAL_LIST = os.getenv("EVAL_LIST", "")

def get_model_step(f):
    match = re.search(r'model_(\d+)\.pt', f)
    return int(match.group(1)) if match else -1

all_models = sorted([(f, get_model_step(f)) for f in os.listdir(eval_dir) if f.endswith('.pt')], key=lambda x: x[1])

if EVAL_LIST:
    # Support both comma and colon as delimiters due to Slurm --export issues
    delimiter = ":" if ":" in EVAL_LIST else ","
    target_steps = set(int(s.strip()) for s in EVAL_LIST.split(delimiter) if s.strip())
    filtered = [f for f, step in all_models if step in target_steps]
    filtered.sort(key=get_model_step)
else:
    filtered = [f for f, step in all_models if EVAL_START <= step <= EVAL_END and step % EVAL_STRIDE == 0]
    if all_models:
        last_f, last_step = all_models[-1]
        if EVAL_START <= last_step <= EVAL_END and last_f not in filtered:
            filtered.append(last_f)
        filtered.sort(key=get_model_step)

eval_files = [filtered[-1]] if EVAL_LAST_ONLY and filtered else filtered
print_info(f"Checkpoints to evaluate: {len(eval_files)} (list={EVAL_LIST}, range=[{EVAL_START}, {EVAL_END}])")
# --- ABLATION LOGIC END ---

eval_logs_dir = os.path.join(fl.logs_dir, 'eval')
os.makedirs(eval_logs_dir, exist_ok=True)
completed_file = os.path.join(eval_logs_dir, '_eval_checkpoint.txt')
completed = set()
if os.path.exists(completed_file):
    with open(completed_file) as f:
        completed = set(line.strip() for line in f if line.strip())

pending_files = [f for f in eval_files if f not in completed]
overall_start = time()
models_count = len(pending_files)

for step, model_file in enumerate(pending_files):
    print_info(f"- Evaluation {step + 1}/{models_count}", flush=True)
    checkpoint = torch.load(os.path.join(eval_dir, model_file), weights_only=False)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    evaluate_benchmarks(model, eval_data)
    with open(completed_file, 'a') as f:
        f.write(model_file + '\n')
