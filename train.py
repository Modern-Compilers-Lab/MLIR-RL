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
    value_update,
    evaluate_benchmark
)

# Set target device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.set_grad_enabled(False)

# Set environments
env = Env()
eval_env = Env(is_inference_env=True)

# Set model
model = Model()

value_optimizer = torch.optim.Adam(
    model.value_network.parameters(),
    lr=cfg.lr
)

ppo_optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)

# Set neptune logs if enabled
neptune_logs = init_neptune(['ppo'] + cfg.tags) if cfg.logging else None

# Start training
for step in range(cfg.nb_iterations):
    print_info(f"--- Iteration: {step}/{cfg.nb_iterations} ---")
    print_info('- Collecting trajectory -')
    trajectory = collect_trajectory(
        model,
        env,
        device,
        neptune_logs
    )

    value_update(
        trajectory,
        model,
        value_optimizer,
        device,
        neptune_logs
    )

    if (step + 1) % 5 == 0:
        ppo_update(
            trajectory,
            model,
            ppo_optimizer,
            device,
            neptune_logs
        )

        print_info('- Evaluating benchmark -')
        evaluate_benchmark(
            model,
            eval_env,
            device,
            neptune_logs
        )

        if cfg.logging:
            neptune_logs["params"].upload_files(['models/ppo_model.pt'])


# Stop logs if enabled
if cfg.logging:
    neptune_logs.stop()
