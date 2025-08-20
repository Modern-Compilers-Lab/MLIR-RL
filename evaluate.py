# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
import os
import torch
from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular import config as cfg, device
from rl_autoschedular.ppo import evaluate_benchmarks
from rl_autoschedular.benchmarks import Benchmarks
from utils.log import print_info, print_success
from time import time
import random
import string
import json
import datetime


# Initialize dask in order to allocate jobs
dm = DaskManager()

# Setup torch
torch.set_grad_enabled(False)
torch.set_num_threads(4)

# Load data
eval_data = dm.load_eval_data(Benchmarks(is_training=False))

# Prepare logging
fl = FileLogger()
print_info(f"Config: {cfg}")
print_success(f'Logging to: {fl.run_dir}')

# Prepare the temporary execution database
random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
tmp_exec_data_file = f'tmp-debug/exec/{random_str}.json' if cfg.debug else f'tmp/exec/{random_str}.json'
if not os.path.exists(tmp_exec_data_file):
    os.makedirs(os.path.dirname(tmp_exec_data_file), exist_ok=True)
    with open(tmp_exec_data_file, "w") as file:
        json.dump({}, file)
print_info(f"Temporary execution data saved to: {tmp_exec_data_file}")

if cfg.exec_data_file:
    print_info(f"Global execution data located in: {cfg.exec_data_file}")

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

time_ms = 0
eta = 0
models_count = len(eval_files)
for step, model_file in enumerate(eval_files):
    print_info(f"- Evaluation {step + 1}/{models_count} ({100 * (step + 1) / models_count:.2f}%) ({time_ms}ms) < ({eta})")

    main_start = time()

    model_path = os.path.join(eval_dir, model_file)
    if not os.path.exists(model_path):
        print_info(f"Model file {model_path} does not exist. Skipping.")
        continue
    model.load_state_dict(torch.load(model_path, weights_only=True))

    evaluate_benchmarks(model, eval_data, tmp_exec_data_file)

    main_end = time()
    time_ms = int((main_end - main_start) * 1000)
    eta = datetime.timedelta(seconds=time_ms * (models_count - step - 1) / 1000)
