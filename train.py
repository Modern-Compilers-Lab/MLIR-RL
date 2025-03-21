# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
import torch
from tqdm import trange
from rl_autoschedular import config as cfg
from utils.log import print_info, print_success
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
# torch.autograd.set_detect_anomaly(True)
torch.set_num_threads(4)

print_info(f"Config: {cfg}")

# Set neptune logs if enabled
neptune_logs = init_neptune(
    tags=['ppo'] + cfg.tags,
    mode='debug' if cfg.debug else 'sync'
)
print_success("Neptune initialized")

# Set environments
env = Env(log_schedule=cfg.debug)
eval_env = Env(
    tmp_file=env.tmp_file,
    inference_env=True,
)
print_success("Environments initialized")

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

        if not cfg.debug:
            neptune_logs["params"].upload_files(['models/ppo_model.pt'])


# Stop logs if enabled
if not cfg.debug:
    neptune_logs.stop()
