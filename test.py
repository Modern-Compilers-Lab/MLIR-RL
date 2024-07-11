from utils.hiearchy_simple_ppo_env import ParallelEnv
from utils.ppo_model import HiearchyModel as MyModel
from utils.consts import PPO_ACTIONS
from utils.consts import (
    MAX_NUM_STORES_LOADS,
    MAX_NUM_LOOPS,
    MAX_NUM_LOAD_STORE_DIM
)

import torch
import numpy as np

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def evaluate_benchamrk(model, env, logs):

    for i, operation in enumerate(env.env.operations_files):
        
        print(f'Operation ({i}):', env.env.operations_files[i][0])
        
        # Reset the environement with the specific operation
        state, obs = env.reset(i)
        obs = torch.cat(obs).to(device)

        while True:

            with torch.no_grad():
                # Select the action using the model
                action, action_log_p, values, entropy = model.sample(obs)

            # Apply the action and get the next state
            next_obs, reward, terminated, truncated, next_state, final_state = env.step(state, action, model)
            
            done = terminated[0] or truncated[0]
            if done:
                final_state = final_state[0]
                speedup_metric = final_state.root_exec_time /  final_state.exec_time
                print('Base execution time:', final_state.root_exec_time / 1e9, 'ms')
                print('New execution time:', final_state.exec_time / 1e9, 'ms')
                print('speedup:', speedup_metric)
                break             

            state = next_state
            obs = torch.cat(next_obs).to(device)

        print('\n\n\n')



CONFIG = {
    'len_trajectory': 64,
    'ppo_batch_size': 64,
    'steps':10000,
    'ppo_epochs':4,
    'logs':True,
    'entropy_coef':0.01,
    'lr':0.001,
    'truncate':5,
    'json_file':"generated_data/bigger_input_nn_(32x230x230x3)operations.json",
}

env = ParallelEnv(
    json_file=CONFIG["json_file"],
    num_env=1,
    truncate=CONFIG["truncate"],
    reset_repeat=1,
    step_repeat=1,
)



# Define the model:
input_dim = MAX_NUM_LOOPS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM*MAX_NUM_STORES_LOADS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM + 5 + \
    MAX_NUM_LOOPS*3*CONFIG["truncate"]
print('input_dim:', input_dim)

model = MyModel(
    input_dim=input_dim,
    num_loops=MAX_NUM_LOOPS
)

# model.load_state_dict(torch.load('models/demo_model_bigger_nn.pt'))
model.load_state_dict(torch.load('models/demo_ppo_model.pt'))
        

evaluate_benchamrk(
    env=env,
    model=model,
    logs=logs
)