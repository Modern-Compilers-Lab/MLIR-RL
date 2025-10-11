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
from rl_autoschedular.ppo import evaluate_benchmarks

# Import RL components
from rl_autoschedular.model import HiearchyModel as Model
import time

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))



print_info(f"Config: {cfg}")
print_success(f'Logging to: {fl.run_dir}')

# Set environments
eval_env = Env(is_training=False,run_name="ppo_online_eval")
print_success(f"Environments initialized: {eval_env.tmp_file}")

# Set model
model_chkpt = "./checkpoints/model.pth"
model = Model().to(device)
checkpoint = torch.load(model_chkpt, map_location="cpu")
model.load_state_dict(checkpoint, strict=False)  # allow partial load
model.eval()

env_time = evaluate_benchmarks(
    model,
    eval_env,
    step=1
)

print(env_time)