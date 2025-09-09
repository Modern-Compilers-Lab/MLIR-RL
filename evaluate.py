# Load environment variables
from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

# Import modules
import os
import logging
import torch
from typing import Optional
from rl_autoschedular.execution import Execution
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular import device
from rl_autoschedular.ppo import evaluate_benchmarks
from rl_autoschedular.benchmarks import Benchmarks
from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger
from utils.config import Config
from utils.log import print_info, print_success
from datetime import timedelta
from time import time

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
main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

# Initialize execution singleton
Execution(fl.exec_data_file, main_exec_data)

# Prepare logging
print_info(f"Config: {cfg}")
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
    raise ValueError("EVAL_DIR environment variable is not set.")
eval_dir = os.path.abspath(eval_dir)

# Read the files in the evaluation directory
eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.pt')]

# Order files
eval_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

iter_time_dlt = 0
elapsed_dlt = 0
eta_dlt = 0
overall_start = time()
models_count = len(eval_files)
for step, model_file in enumerate(eval_files):
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
    model.load_state_dict(torch.load(model_path, weights_only=True))

    evaluate_benchmarks(model, eval_data)

    main_end = time()
    iter_time = main_end - main_start
    elapsed = main_end - overall_start
    eta = elapsed * (cfg.nb_iterations - step - 1) / (step + 1)
    iter_time_dlt = timedelta(seconds=iter_time)
    elapsed_dlt = timedelta(seconds=int(elapsed))
    eta_dlt = timedelta(seconds=int(eta))
