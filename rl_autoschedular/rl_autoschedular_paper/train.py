"""Main training script for MLIR RL using PPO.

This module implements the primary training loop for the reinforcement learning system.
It initializes the models, loads benchmark data, and iterates through PPO training steps
including trajectory collection, policy updates, and periodic evaluation.

Execution Command:
    ```bash
    CONFIG_FILE_PATH=path/to/config.json poetry run train
    ```
"""

from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

import os
import json
import random
from typing import Optional
from time import time
from datetime import timedelta
import torch
import numpy as np
from rl_autoschedular_paper.benchmarks import Benchmarks
from rl_autoschedular_paper.execution import Execution
from rl_autoschedular_paper.model import HiearchyModel as Model
from rl_autoschedular_paper import device
from rl_autoschedular_paper.trajectory import TrajectoryData
from rl_autoschedular_paper.ppo import collect_trajectory, ppo_update, value_update, evaluate_benchmarks
from rl_autoschedular_paper.utils.log import print_info, print_success
from rl_autoschedular_paper.utils.config import Config
from rl_autoschedular_paper.utils.dask_manager import DaskManager
from rl_autoschedular_paper.utils.file_logger import FileLogger
from rl_autoschedular_paper.utils.gpu_occupier import GPUOccupier


def main():
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize singleton classes
    cfg = Config()
    fl = FileLogger()
    dm = DaskManager()
    go = GPUOccupier()

    # Start GPU Occupier
    if device.type == "cuda":
        go.start(device)

    # Data loading
    def load_train_data():
        return Benchmarks()

    def load_eval_data():
        return Benchmarks(is_training=False)

    def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
        main_exec_data = None
        if Config().main_exec_data_file:
            with open(Config().main_exec_data_file) as f:
                main_exec_data = json.load(f)
        return main_exec_data

    train_data = dm.run_and_register_to_workers(load_train_data)
    eval_data = dm.run_and_register_to_workers(load_eval_data)
    main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

    # Initialize execution singleton
    Execution(fl.exec_data_file, main_exec_data)

    print_success(f'Logging to: {fl.run_dir}')
    if cfg.main_exec_data_file:
        print_info(f"Global execution data located in: {cfg.main_exec_data_file}")

    # Setup torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    if cfg.debug:
        torch.autograd.set_detect_anomaly(True)

    # Initiate model
    with go.gpu_needed():
        model = Model().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr
    )
    print_success("Model initialized")

    # Start training
    old_trajectory: Optional[TrajectoryData] = None
    iter_time_dlt = 0
    elapsed_dlt = 0
    eta_dlt = 0
    overall_start = time()
    for step in range(cfg.nb_iterations):
        print_info(
            f"- Main Loop {step + 1}/{cfg.nb_iterations}"
            f" ({100 * (step + 1) / cfg.nb_iterations:.2f}%)"
            f" ({iter_time_dlt}/it) ({elapsed_dlt} < {eta_dlt})",
            flush=True
        )

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
            with go.gpu_needed():
                value_update(trajectory, model, optimizer)

        # Update policy model with PPO
        with go.gpu_needed():
            ppo_update(trajectory, model, optimizer)

        # Save model + optimizer state every 50 iterations
        if step % 50 == 0:
            torch.save(
                {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step},
                os.path.join(fl.models_dir, f'model_{step}.pt')
            )

        if (step + 1) % 100 == 0:
            print_info('- Evaluating benchmarks -')
            evaluate_benchmarks(model, eval_data)

        main_end = time()
        iter_time = main_end - main_start
        elapsed = main_end - overall_start
        eta = elapsed * (cfg.nb_iterations - step - 1) / (step + 1)
        iter_time_dlt = timedelta(seconds=iter_time)
        elapsed_dlt = timedelta(seconds=int(elapsed))
        eta_dlt = timedelta(seconds=int(eta))

    if (step + 1) % 100 != 0:
        print_info('- Evaluating benchmarks -')
        evaluate_benchmarks(model, eval_data)

    # Save final model
    torch.save(
        {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step},
        os.path.join(fl.models_dir, f'model_{step + 1}.pt')
    )

    dm.close()
    go.stop()


if __name__ == "__main__":
    main()
