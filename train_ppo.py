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
    evaluate_benchmarks
)

import time
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))

if cfg.debug:
    torch.autograd.set_detect_anomaly(True)

print_info(f"Config: {cfg}")
print_success(f'Logging to: {fl.run_dir}')

# Set environments

# run_name for /tmp/ path
env = Env(is_training=True,run_name="online_ppo")
eval_env = Env(is_training=False,run_name="online_ppo")
print_success(f"Environments initialized: {env.tmp_file}")

# Set model
model = Model().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4
)
print_success("Model initialized")

train_start = time.perf_counter()
total_env_time = 0.0
total_eval_time = 0.0

# Start training
for step in range(cfg.nb_iterations):
    print_info(f"- Main Loop {step + 1}/{cfg.nb_iterations} ({100 * (step + 1) / cfg.nb_iterations:.2f}%)")
    trajectory , env_time = collect_trajectory(
        model,
        env,
        step,
    )
    total_env_time += env_time
    
    

    # Fit value model to trajectory rewards
    if cfg.value_epochs > 0:
        value_update(
            trajectory,
            model,
            optimizer,
            step
        )

    ppo_update(
        trajectory,
        model,
        optimizer,
        step
    )

    if (step + 1) % 50 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                env.tmp_file.replace('.mlir', ''),
                f'model_{step}.pth'
            )
        )

    if (step + 1) % 50 == 0:
        start_eval = time.perf_counter()
        print_info('- Evaluating benchmark -')
        eval_time = evaluate_benchmarks(
            model,
            eval_env,
            step
        )
        end_eval = time.perf_counter()
        total_eval_time += end_eval - start_eval
train_end = time.perf_counter()
total_train_time = train_end - train_start - total_eval_time
print_success(f"- Training completed in {total_train_time:.2f} seconds")
print_success(f"- Evaluation completed in {total_eval_time:.2f} seconds")
print_success(f"- Total environment time: {total_env_time:.2f} seconds")
print_success(f"- Total eval env time: {total_eval_time:.2f} seconds")
print_success(f"- Percentage of time in environment: {100 * total_env_time / total_train_time:.2f}%")