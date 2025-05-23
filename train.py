# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
import torch
import os
from tqdm import trange
from rl_autoschedular import config as cfg
from rl_autoschedular import file_logger as fl
from utils.log import print_info, print_success
from rl_autoschedular.ppo import (
    collect_trajectory,
    ppo_update,
    value_update,
    update_trajectory_values,
    evaluate_benchmark
)

# Set target device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_grad_enabled(False)
torch.set_num_threads(4)
if cfg.debug:
    torch.autograd.set_detect_anomaly(True)

print_info(f"Config: {cfg}")
print_success(f'Logging to: {fl.run_dir}')

# Set environments
env = Env(is_training=True)
eval_env = Env(is_training=False, tmp_file=env.tmp_file)
print_success(f"Environments initialized: {env.tmp_file}")

# Set model
model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)
print_success("Model initialized")

# Start training
# ppo_trajectory = None
for step in trange(cfg.nb_iterations, desc='Main loop'):
    trajectory = collect_trajectory(
        model,
        env,
        step,
        device,
    )

    # Fit value model to trajectory rewards
    if cfg.value_epochs > 0:
        value_update(
            trajectory,
            model,
            optimizer,
            device,
        )

        # Update trajectory values to be used in PPO update
        update_trajectory_values(
            trajectory,
            model,
            device,
        )

    ppo_update(
        trajectory,
        model,
        optimizer,
        device,
    )

    if (step + 1) % 5 == 0:
        torch.save(
            model.state_dict(),
            os.path.join(
                env.tmp_file.replace('.mlir', ''),
                f'model_{step}.pth'
            )
        )

print_info('- Evaluating benchmark -')
evaluate_benchmark(
    model,
    eval_env,
    device,
)
