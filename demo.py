
# Setup environment
from dotenv import load_dotenv
load_dotenv(override=True)


# Import modules
import torch
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.ppo import evaluate_benchmark


# Configure torch
torch.set_grad_enabled(False)
torch.set_num_threads(4)


# Instantiate the environment
eval_env = Env(is_training=False)


# Load the model
model_path = "models/model.pth"
model = Model()
model.load_state_dict(torch.load(model_path, weights_only=True))


# Evaluate the model
evaluate_benchmark(
    model,
    eval_env,
)
