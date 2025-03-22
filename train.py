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

# Set environments
env = Env()
eval_env = Env(tmp_file=env.tmp_file)
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
        device,
    )

    ppo_update(
        trajectory,
        model,
        optimizer,
        device,
    )

    if (step + 1) % 5 == 0:
        print_info('- Evaluating benchmark -')
        evaluate_benchmark(
            model,
            eval_env,
            device,
        )
