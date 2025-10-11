# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)


import torch
import os
from typing import Optional
from utils.log import print_info, print_success

# Import environment
from rl_autoschedular.env import Env

# config, file_logger, device
from rl_autoschedular import config as cfg, file_logger as fl, device

# Import RL components
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.trajectory import TrajectoryData
from rl_autoschedular.ppo import (
    collect_trajectory,
    ppo_update,
    value_update,
    evaluate_benchmark
)


torch.set_grad_enabled(False)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
if cfg.debug:
    torch.autograd.set_detect_anomaly(True)

print_info(f"Config: {cfg}")
print_success(f'Logging to: {fl.run_dir}')

# Set environments
env = Env(is_training=True)
env.save_benchmarks_data_to_json("my_benchmarks.json")

eval_env = Env(is_training=False, tmp_file=env.tmp_file)

print_success(f"Environments initialized: {env.tmp_file}")