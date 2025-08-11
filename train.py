# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
import torch
import os
from rl_autoschedular import config as cfg, file_logger as fl, device
from rl_autoschedular.trajectory import TrajectoryData
from utils.log import print_info, print_success
from typing import Optional
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
eval_env = Env(is_training=False, tmp_file=env.tmp_file)
print_success(f"Environments initialized: {env.tmp_file}")

# Set model
model = Model().to(device)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)
print_success("Model initialized")

# Start training
old_trajectory: Optional[TrajectoryData] = None
for step in range(cfg.nb_iterations):
    print_info(f"- Main Loop {step + 1}/{cfg.nb_iterations} ({100 * (step + 1) / cfg.nb_iterations:.2f}%)")
    trajectory = collect_trajectory(
        model,
        env,
        step,
    )

    # Extend trajectory with previous trajectory
    if cfg.reuse_experience:
        if old_trajectory is not None:
            trajectory = old_trajectory + trajectory
        old_trajectory = trajectory.copy()

    # Fit value model to trajectory rewards
    if cfg.value_epochs > 0:
        value_update(
            trajectory,
            model,
            optimizer,
        )

    ppo_update(
        trajectory,
        model,
        optimizer,
    )

    if (step + 1) % 5 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                env.tmp_file.replace('.mlir', ''),
                f'model_{step}.pth'
            )
        )

    if (step + 1) % 1000 == 0:
        print_info('- Evaluating benchmark -')
        evaluate_benchmark(
            model,
            eval_env,
        )

print_info('- Evaluating benchmark -')
evaluate_benchmark(
    model,
    eval_env,
)
