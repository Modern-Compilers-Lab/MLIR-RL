"""Evaluation script for trained MLIR RL models.

This module evaluates all trained model checkpoints found in the `models`
directory on benchmark datasets, measuring optimization quality through
speedup factors and execution times.

Execution Command:
    ```bash
    CONFIG_FILE_PATH=path/to/config.json poetry run evaluate
    ```
"""

from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

import os
import signal
import torch


def _sigabrt_handler(signum, frame):
    raise RuntimeError("MLIR native code crashed (SIGABRT) — caught and continuing")
signal.signal(signal.SIGABRT, _sigabrt_handler)
import numpy as np
import random
import re
from typing import Optional
from rl_autoschedular_paper_transformer.execution import Execution
from rl_autoschedular_paper_transformer.model import HiearchyModel as Model
from rl_autoschedular_paper_transformer import device
from rl_autoschedular_paper_transformer.ppo import evaluate_benchmarks
from rl_autoschedular_paper_transformer.benchmarks import Benchmarks
from rl_autoschedular_paper_transformer.utils.dask_manager import DaskManager
from rl_autoschedular_paper_transformer.utils.gpu_occupier import GPUOccupier
from rl_autoschedular_paper_transformer.utils.file_logger import FileLogger
from rl_autoschedular_paper_transformer.utils.config import Config
from rl_autoschedular_paper_transformer.utils.log import print_info, print_success
from datetime import timedelta
from time import time

eval_dir = 'models'


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
    def load_eval_data():
        return Benchmarks(is_training=False)

    def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
        return None

    eval_data = dm.run_and_register_to_workers(load_eval_data)
    main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

    # Initialize execution singleton
    Execution(fl.exec_data_file, main_exec_data)

    # Setup signal handler again (MLIR imports / initialization overrides Python's signal handler)
    signal.signal(signal.SIGABRT, _sigabrt_handler)

    # Prepare logging
    print_success(f'Logging to: {fl.run_dir}')

    # Setup torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)

    # Initiate model
    with go.gpu_needed():
        model = Model().to(device)
    print_success("Model initialized")

    # Read the files in the evaluation directory
    eval_files = [f for f in os.listdir(eval_dir) if re.match(r'model_\d+\.(pt|pth)', f)]

    # Order files
    eval_files.sort(key=lambda x: int(re.match(r'model_(\d+)\.(pt|pth)', x).group(1)))

    iter_time_dlt = 0
    elapsed_dlt = 0
    eta_dlt = 0
    overall_start = time()
    models_count = len(eval_files)
    for step, model_file in enumerate(eval_files):
        print_info(
            f"- Evaluation {model_file}"
            f" ({100 * (step + 1) / models_count:.2f}%)"
            f" ({iter_time_dlt}/it) ({elapsed_dlt} < {eta_dlt})",
            flush=True
        )

        main_start = time()

        model_path = os.path.join(eval_dir, model_file)
        if not os.path.exists(model_path):
            print_info(f"Model file {model_path} does not exist. Skipping.")
            continue
        with go.gpu_needed():
            model.load_state_dict(torch.load(model_path, weights_only=True))

        evaluate_benchmarks(model, eval_data)

        main_end = time()
        iter_time = main_end - main_start
        elapsed = main_end - overall_start
        eta = elapsed * (cfg.nb_iterations - step - 1) / (step + 1)
        iter_time_dlt = timedelta(seconds=iter_time)
        elapsed_dlt = timedelta(seconds=int(elapsed))
        eta_dlt = timedelta(seconds=int(eta))


if __name__ == "__main__":
    main()
