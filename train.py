# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
import torch
from rl_autoschedular import config as cfg
from utils.log import print_info
from utils.neptune_utils import init_neptune
from rl_autoschedular.ppo import (
    collect_trajectory,
    ppo_update,
    evaluate_benchmark
)

# Set target device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_grad_enabled(False)
torch.autograd.set_detect_anomaly(True)

print_info(f"Config: {cfg}")

# Set environments
env = Env()
eval_env = Env(env.tmp_file, log_schedule=True)

# Set model
model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)

# Set neptune logs if enabled
neptune_logs = init_neptune(
    tags=['ppo'] + cfg.tags,
    mode='sync' if cfg.logging else 'debug'
)

# Start training
# ppo_trajectory = None
for step in range(cfg.nb_iterations):
    print_info(f"--- Iteration: {step}/{cfg.nb_iterations} {step/cfg.nb_iterations*100:.2f}% ---")
    print_info('- Collecting trajectory -')
    trajectory = collect_trajectory(
        model,
        env,
        step,
        neptune_logs,
        device,
    )

    ppo_update(
        trajectory,
        model,
        optimizer,
        neptune_logs,
        device,
    )

    if (step + 1) % 5 == 0:
        print_info('- Evaluating benchmark -')
        evaluate_benchmark(
            model,
            eval_env,
            neptune_logs,
            device,
        )

        if cfg.logging:
            neptune_logs["params"].upload_files(['models/ppo_model.pt'])


# Stop logs if enabled
if cfg.logging:
    neptune_logs.stop()
