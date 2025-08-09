# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)

# Import modules
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
import torch
import os
from tqdm import tqdm
from rl_autoschedular import config as cfg
from rl_autoschedular import file_logger as fl
from utils.log import print_info, print_success
from rl_autoschedular.ppo import evaluate_benchmark

torch.set_grad_enabled(False)
torch.set_num_threads(4)

print_info(f"Config: {cfg}")
print_success(f'Logging to: {fl.run_dir}')

# Set environments
eval_env = Env(is_training=False)
print_success(f"Environments initialized: {eval_env.tmp_file}")

# Start training
eval_dir = os.getenv('EVAL_DIR')
if eval_dir is None:
    raise ValueError("EVAL_DIR environment variable is not set.")
eval_dir = os.path.abspath(eval_dir)

# Read the files in the evaluation directory
eval_files = [f for f in os.listdir(eval_dir) if f.endswith('.pth')]

# Order files
eval_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

files_tqdm = tqdm(eval_files, desc='Evaluating models')
for model_file in files_tqdm:
    files_tqdm.set_postfix_str(f"Evaluating {model_file}")
    model = Model()
    model_path = os.path.join(eval_dir, model_file)
    if not os.path.exists(model_path):
        print_info(f"Model file {model_path} does not exist. Skipping.")
        continue
    model.load_state_dict(torch.load(model_path, weights_only=True))
    evaluate_benchmark(
        model,
        eval_env,
    )
