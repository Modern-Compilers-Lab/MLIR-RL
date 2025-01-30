# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import ParallelEnv
from rl_autoschedular.model import HiearchyModel as Model
import torch
from tqdm import tqdm
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

print_info('Finish imports')

# Set environments
env = ParallelEnv(
    num_env=1,
    reset_repeat=1,
    step_repeat=1
)
eval_env = ParallelEnv(
    num_env=1,
    reset_repeat=1,
    step_repeat=1
)
print_info('Env build ...')
# NOTE: using only one environment
print_info(f'tmp_file = {env.envs[0].tmp_file}')

# Print configuration
print_info('Configuration:')
print_info(cfg)

# Set model
model = Model()
print_info('input_dim:', model.input_dim)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.lr
)

# Set neptune logs if enabled
neptune_logs = init_neptune(['hierchical', 'sparse_reward'] + cfg.tags) if cfg.logging else None

# Start training
print_info('Start training ... ')
tqdm_range = tqdm(range(cfg.nb_iterations), desc='Main loop')
for step in tqdm_range:

    trajectory = collect_trajectory(
        cfg.len_trajectory,
        model,
        env,
        device=device,
        neptune_logs=neptune_logs
    )

    loss = ppo_update(
        trajectory,
        model,
        optimizer,
        ppo_epochs=cfg.ppo_epochs,
        ppo_batch_size=cfg.ppo_batch_size,
        device=device,
        entropy_coef=cfg.entropy_coef,
        neptune_logs=neptune_logs
    )

    torch.save(model.state_dict(), 'models/ppo_model.pt')

    if step % 5 == 0:
        evaluate_benchmark(
            model=model,
            env=eval_env,
            device=device,
            neptune_logs=neptune_logs
        )

        if cfg.logging:
            neptune_logs["params"].upload_files(['models/ppo_model.pt'])


# Stop logs if enabled
if cfg.logging:
    neptune_logs.stop()

print_info('Training ended ... ')
