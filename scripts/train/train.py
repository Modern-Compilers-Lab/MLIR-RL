from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)
load_dotenv('.env.debug')

import os
import logging
import signal
import torch
import json

# Catch SIGABRT from MLIR native code — convert to Python exception
def _sigabrt_handler(signum, frame):
    raise RuntimeError("MLIR native code crashed (SIGABRT) — caught and continuing")
signal.signal(signal.SIGABRT, _sigabrt_handler)
from utils.log import print_info, print_success
from utils.config import Config
from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger
from utils.implementation import get_autoschedular_impl, import_autoschedular_module
from typing import Optional
from time import time
from datetime import timedelta
import shutil

AUTOSCHEDULER_IMPL = get_autoschedular_impl()
Benchmarks = import_autoschedular_module("benchmarks", AUTOSCHEDULER_IMPL).Benchmarks
Execution = import_autoschedular_module("execution", AUTOSCHEDULER_IMPL).Execution
Model = import_autoschedular_module("model", AUTOSCHEDULER_IMPL).HiearchyModel
device = import_autoschedular_module("", AUTOSCHEDULER_IMPL).device
TrajectoryData = import_autoschedular_module("trajectory", AUTOSCHEDULER_IMPL).TrajectoryData
ppo_module = import_autoschedular_module("ppo", AUTOSCHEDULER_IMPL)
collect_trajectory = ppo_module.collect_trajectory
ppo_update = ppo_module.ppo_update
value_update = ppo_module.value_update

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

# Safety: refuse fresh training if experiment already has results
resume_from = os.getenv("RESUME_FROM")
force_new = os.getenv("FORCE_NEW")
if (resume_from is None and force_new is None
        and os.path.isfile(fl.train_results_file)
        and os.path.getsize(fl.train_results_file) > 2):
    raise RuntimeError(
        f"Experiment at {fl.run_dir} already has training results. "
        "Use --resume to continue training, or set FORCE_NEW=1 to overwrite."
    )

# Clear per-iteration log files on fresh start to prevent accumulation
if resume_from is None:
    fl.clear_per_iter_logs()


# Data loading
def load_train_data():
    return Benchmarks()




def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
    main_exec_data = None
    if Config().main_exec_data_file:
        with open(Config().main_exec_data_file) as f:
            main_exec_data = json.load(f)
    return main_exec_data


train_data = dm.run_and_register_to_workers(load_train_data)
main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

# Initialize execution singleton
Execution(fl.exec_data_file, main_exec_data)

# Setup signal handler again (MLIR imports / initialization overrides Python's signal handler)
signal.signal(signal.SIGABRT, _sigabrt_handler)

print_info(f"Config: {cfg}")
print_info(f"Autoscheduler implementation: {AUTOSCHEDULER_IMPL}")
print_success(f'Logging to: {fl.run_dir}')
if cfg.main_exec_data_file:
    print_info(f"Global execution data located in: {cfg.main_exec_data_file}")

# Setup torch
torch.set_grad_enabled(False)
torch.set_num_threads(4)
if cfg.debug:
    torch.autograd.set_detect_anomaly(True)

# Initiate model
model = Model().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)
print_success("Model initialized")

# Check for --resume flag (external checkpoint path)
start_step = 1
resume_from = os.getenv("RESUME_FROM")
if resume_from:
    import re
    import builtins
    resume_models_dir = os.path.join(resume_from, "models")
    if os.path.isdir(resume_models_dir):
        resume_models_list = [
            f for f in os.listdir(resume_models_dir)
            if f.startswith('model_') and f.endswith('.pt')
        ]
        if resume_models_list:
            latest_model_file = builtins.max(
                resume_models_list,
                key=lambda f: int(re.search(r'model_(\d+)\.pt', f).group(1))
            )
            latest_step = int(re.search(r'model_(\d+)\.pt', latest_model_file).group(1))
            checkpoint = torch.load(
                os.path.join(resume_models_dir, latest_model_file),
                map_location=device,
                weights_only=False
            )
            model.load_state_dict(checkpoint['model'])
            try:
                optimizer.load_state_dict(checkpoint['optimizer'])
                print_success(
                    f"Resumed model + optimizer from --resume {resume_from}/models/{latest_model_file}"
                )
            except Exception:
                print_success(
                    f"Resumed model from --resume {resume_from}/models/{latest_model_file} (no optimizer)"
                )
            start_step = latest_step + 1
        else:
            print_info(f"Warning: --resume {resume_from} has no model files in models/")
    else:
        print_info(f"Warning: --resume {resume_from}/models/ directory not found")

print_info(f"Training loop: start={start_step}, end={cfg.nb_iterations}, steps={cfg.nb_iterations - start_step}")

# Start training
old_trajectory: Optional[TrajectoryData] = None
iter_time_dlt = 0
elapsed_dlt = 0
eta_dlt = 0
overall_start = time()
for step in range(start_step, cfg.nb_iterations):
    print_info(
        f"- Main Loop {step + 1}/{cfg.nb_iterations}"
        f" ({100 * (step + 1) / cfg.nb_iterations:.2f}%)"
        f" ({iter_time_dlt}/it) ({elapsed_dlt} < {eta_dlt})",
        flush=True
    )

    try:

        main_start = time()

        # Collect trajectory using the model
        trajectory = collect_trajectory(train_data, model, step)

        # Extend trajectory with previous trajectory
        if cfg.reuse_experience != 'none':
            reuse_start = time()
            if old_trajectory is not None:
                trajectory = old_trajectory + trajectory
            old_trajectory = trajectory.copy()
            reuse_end = time()
            reuse_time_ms = int((reuse_end - reuse_start) * 1000)
            print_info(f"Reuse time: {reuse_time_ms}ms")

        # Fit value model to trajectory rewards
        if cfg.value_epochs > 0:
            value_update(trajectory, model, optimizer)

        # Update policy model with PPO
        ppo_update(trajectory, model, optimizer)

        # Save model + optimizer state every 50 iterations
        if step % 50 == 0:
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step},
                os.path.join(fl.models_dir, f'model_{step}.pt')
            )

        main_end = time()
        iter_time = main_end - main_start
        elapsed = main_end - overall_start
        eta = elapsed * (cfg.nb_iterations - step - 1) / (step + 1)
        iter_time_dlt = timedelta(seconds=iter_time)
        elapsed_dlt = timedelta(seconds=int(elapsed))
        eta_dlt = timedelta(seconds=int(eta))

    except RuntimeError as e:
        print_info(f"Iteration {step} crashed (MLIR SIGABRT): {e}", flush=True)
        continue

# Save final model
torch.save(
    {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step},
    os.path.join(fl.models_dir, f'model_{step + 1}.pt')
)
