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
from rl_autoschedular.trajectory import TrajectoryData
from rl_autoschedular.ppo import collect_trajectory, ppo_update, value_update, evaluate_benchmarks
from rl_autoschedular.benchmarks import Benchmarks
from utils.log import print_info, print_success
from typing import Optional
from time import time


# Initialize dask in order to allocate jobs
dm = DaskManager()

# Setup torch
torch.set_grad_enabled(False)
torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "4")))
if cfg.debug:
    torch.autograd.set_detect_anomaly(True)

# Load data
train_size = dm.load_train_data(Benchmarks())
eval_size = dm.load_eval_data(Benchmarks(is_training=False))

# Prepare logging
fl = FileLogger()
print_info(f"Config: {cfg}")
print_success(f'Logging to: {fl.run_dir}')

# Initiate model
model = Model().to(device)
torch.save(model.state_dict(), fl.last_model_path)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)
print_success("Model initialized")

# Start training
old_trajectory: Optional[TrajectoryData] = None
time_ms = 0
for step in range(cfg.nb_iterations):
    print_info(f"- Main Loop {step + 1}/{cfg.nb_iterations} ({100 * (step + 1) / cfg.nb_iterations:.2f}%) ({time_ms}ms)")

    start = time()

    # Collect trajectory using the model
    trajectory = collect_trajectory(train_size, step)

    # Extend trajectory with previous trajectory
    if cfg.reuse_experience:
        if old_trajectory is not None:
            trajectory = old_trajectory + trajectory
        old_trajectory = trajectory.copy()

    # Fit value model to trajectory rewards
    if cfg.value_epochs > 0:
        value_update(trajectory, model, optimizer)

    # Update policy model with PPO
    ppo_update(trajectory, model, optimizer)

    # Save the model
    torch.save(model.state_dict(), fl.last_model_path)
    if (step + 1) % 5 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                fl.models_dir,
                f'model_{step}.pt'
            )
        )

    if (step + 1) % 1000 == 0:
        print_info('- Evaluating benchmarks -')
        evaluate_benchmarks(eval_size)

    end = time()
    time_ms = int((end - start) * 1000)

if (step + 1) % 1000 != 0:
    print_info('- Evaluating benchmarks -')
    evaluate_benchmarks(eval_size)
