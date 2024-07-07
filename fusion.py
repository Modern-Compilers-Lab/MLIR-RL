from fusion_utils.fusion_env import ParallelEnv
from utils.ppo_model import PPOSimpleModel as MyModel
from utils.consts import PPO_ACTIONS
from utils.consts import (
    MAX_NUM_STORES_LOADS,
    MAX_NUM_LOOPS,
    MAX_NUM_LOAD_STORE_DIM
)

import torch
import numpy as np

from tqdm import tqdm
import neptune



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_info(*args):
    message = ' '.join(map(str, args))
    print(f"\033[94m[INFO]\t {message}\033[0m")

def print_success(*args):
    message = ' '.join(map(str, args))
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")

def print_alert(*args):
    message = ' '.join(map(str, args))
    print(f"\033[93m[ALERT]\t {message}\033[0m")

def print_error(*args):
    message = ' '.join(map(str, args))
    print(f"\033[91m[ERROR]\t {message}\033[0m")


print_info('Finish imports')

def init_neptune(tags: list):
    tags = list(map(str, tags))
    run = neptune.init_run(
        project="nazim-bendib/mlir-rl",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxNDVjNWJkYi1mMTIwLTRmNDItODk3Mi03NTZiNzIzZGNhYzMifQ==",
        tags=tags,
        # mode="sync"
    ) 
    return run






def collect_trajectory(len_trajectory, model, logs=False):

    batch_state, batch_obs = env.reset()
    batch_obs = [obs.to(device) for obs in batch_obs]

    stored_state = []
    stored_action_index = []
    stored_value = []
    stored_action_log_p = []
    stored_x = []
    stored_reward = []
    stored_done = []

    # for i in tqdm(range(len_trajectory)):
    for i in range(len_trajectory):

        x = torch.cat(batch_obs)
        with torch.no_grad():
            action_index, action_log_p, values = model.sample(x)
            new_action_log_p, new_entropy, new_values = model.get_p(x, action_index)
            assert (new_action_log_p == action_log_p).all(), 'check the get_p yerham babak'
            
        

        batch_next_obs, batch_reward, batch_terminated, batch_truncated, batch_next_state, batch_final_state = env.step(batch_state, action_index, model)
                
        stored_action_index += action_index
        
        stored_state.append(batch_state[0])
        stored_value.append(values)
        stored_action_log_p.append(action_log_p)
        stored_x.append(x)
        stored_reward.append(torch.tensor(batch_reward).unsqueeze(0))
        stored_done.append(torch.tensor(batch_terminated).unsqueeze(0))


        # print(batch_next_state[0].actions)

        for i in range(env.num_env):
            done     = batch_terminated[i] or batch_truncated[i]
            final_state = batch_final_state[i]
            # print(done)
            if done:
                speedup_metric = final_state.root_exec_time /  final_state.exec_time
                print('-'*70)
                print(final_state.operation_id)
                print(final_state.transformation_history)
                print('cummulative_reward:', final_state.cummulative_reward)
                print('speedup:', speedup_metric)
                print('-'*70)
                if logs:
                    neptune_logs['train/final_speedup'].append(speedup_metric)
                    neptune_logs['train/cummulative_reward'].append(final_state.cummulative_reward)
                    neptune_logs[f'train/{final_state.operation_id}_speedup'].append(speedup_metric)
                    
                                


        batch_state = batch_next_state
        batch_obs = batch_next_obs


    with torch.no_grad():
        x = torch.cat(batch_obs) 
        _, _, next_value = model.sample(x)

    
    stored_action_index = stored_action_index
    stored_value = torch.concatenate(stored_value)
    stored_action_log_p = torch.concatenate(stored_action_log_p)
    stored_x = torch.concatenate(stored_x)
    stored_reward = torch.concatenate(stored_reward).float()
    stored_done = torch.concatenate(stored_done).float()    
    
    stored_next_value = torch.concatenate((stored_value[1:], next_value))
    assert (stored_value[1:] == stored_next_value[:-1]).all()
        

    trajectory = (
        stored_state,
        stored_action_index,
        stored_value.detach(),
        stored_next_value.detach(),
        stored_action_log_p.detach(),
        stored_x.detach(),
        stored_reward.detach(),
        stored_done.detach(),
    )

    return trajectory


def shuffle_trajectory(trajectory):

    stored_state, stored_action_index, stored_value, stored_next_value, stored_action_log_p, stored_x, stored_reward, stored_done = trajectory


    permutation = torch.randperm(stored_action_log_p.size()[0])

    stored_state = [stored_state[i] for i in permutation]
    stored_action_index = [stored_action_index[i] for i in permutation]
    stored_value = stored_value[permutation]
    stored_next_value = stored_next_value[permutation]
    stored_action_log_p = stored_action_log_p[permutation]
    stored_x = stored_x[permutation]
    stored_reward = stored_reward[permutation]
    stored_done = stored_done[permutation]

    trajectory = (
        stored_state,
        stored_action_index,
        stored_value,
        stored_next_value,
        stored_action_log_p,
        stored_x,
        stored_reward,
        stored_done
    )

    return trajectory


def shuffle_ppo_data(stored_action_index, stored_action_log_p, stored_x, advantage, returns):

    permutation = torch.randperm(stored_action_log_p.size()[0])

    stored_action_index = [stored_action_index[i] for i in permutation]
    stored_action_log_p = stored_action_log_p[permutation]
    stored_x = stored_x[permutation]
    advantage = advantage[permutation]
    returns = returns[permutation]

    return stored_action_index, stored_action_log_p, stored_x, advantage, returns


def compute_gae(done, rewards, values, next_values, gamma=0.99, lambda_=0.95):
    assert len(values) == len(next_values) == len(rewards) == len(done)

    advantages = torch.zeros(done.shape[0], dtype=torch.float32)
    returns = torch.zeros(done.shape[0], dtype=torch.float32)
    last_advantage = 0
    last_return = 0

    for t in reversed(range(done.shape[0])):
        mask = 1.0 - done[t]
        last_value = next_values[t] * mask
        last_advantage = last_advantage * mask
        last_return = last_return * mask

        delta = rewards[t] + gamma * last_value - values[t]
        last_advantage = delta + gamma * lambda_ * last_advantage
        last_return = rewards[t] + gamma * last_return

        advantages[t] = last_advantage
        returns[t] = last_return

    return advantages, returns


def ppo_update(trajectory, model, optimizer, ppo_epochs, ppo_batch_size, logs=False, entropy_coef=0.01):

    loss_i = 0

    for epoch in range(ppo_epochs):

        stored_state, stored_action_index, stored_value, stored_next_value, stored_action_log_p, stored_x, stored_reward, stored_done = trajectory
        
        len_trajectory = stored_x.shape[0]
        assert len_trajectory % ppo_batch_size == 0
            
        stored_value = stored_value.reshape(-1).detach()
        stored_next_value = stored_next_value.reshape(-1).detach()
        stored_reward = stored_reward.reshape(-1).detach()
        stored_done = stored_done.reshape(-1).detach()
        
        advantage, returns = compute_gae(stored_done, stored_reward, stored_value, stored_next_value)
        
        stored_action_index, stored_action_log_p, stored_x, stored_advantage, stored_returns = shuffle_ppo_data(stored_action_index, stored_action_log_p, stored_x, advantage, returns)
        

        acc_loss = 0
        for i in range(len_trajectory // ppo_batch_size):
                        
            begin, end = i*ppo_batch_size, (i+1)*ppo_batch_size

            action_index = stored_action_index[begin:end]
            action_log_p = stored_action_log_p[begin:end].to(device)
            advantage = stored_advantage[begin:end].to(device)
            returns = stored_returns[begin:end].to(device)
            x = stored_x[begin:end].to(device)

            # New predicition:
            new_action_log_p, entropy, new_values = model.get_p(x, action_index)
            
            

            
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            new_action_log_p, action_log_p, advantage = new_action_log_p.reshape(-1), action_log_p.reshape(-1), advantage.reshape(-1)
            
            ratio = torch.exp(new_action_log_p - action_log_p.detach())
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantage
            policy_loss = - torch.min(surr1, surr2).mean()
            
            
            returns, new_values = returns.reshape(-1), new_values.reshape(-1)

            value_loss = ((returns - new_values)**2).mean()
            

            loss = policy_loss - entropy_coef*entropy + 0.5*value_loss
            
            optimizer.zero_grad()
            loss.backward()
            clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            acc_loss += loss.item()
            loss_i += 1

            # Collecting metircs:
            if logs:
                neptune_logs['train/policy_loss'].append(policy_loss.item())
                neptune_logs['train/value_loss'].append(value_loss.item())
                neptune_logs['train/entropy'].append(entropy.item())
                neptune_logs['train/clip_factor'].append(clip_factor.item())






        # print()
        # print('***'*50)
        # print()

    return acc_loss / loss_i


# Start the training:


CONFIG = {
    'len_trajectory': 64,
    'ppo_batch_size': 64,
    'steps':500,
    'ppo_epochs':4,
    'logs':False,
    'entropy_coef':0.01,
    'lr':0.001,
    'truncate':5,
    'mlir_code':'generated_mlir/bigger_input_nn.mlir'
}

env = ParallelEnv(
    mlir_code=CONFIG["mlir_code"],
    num_env=1,
    truncate=CONFIG["truncate"],
    reset_repeat=1,
    step_repeat=1,
)

batch_state, batch_obs = env.reset()

# exit()



print_info('Env build ...')

print(CONFIG)


input_dim = 2*(MAX_NUM_LOOPS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM*MAX_NUM_STORES_LOADS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM + 5) + len(PPO_ACTIONS)
print_info('input_dim:', input_dim)

model = MyModel(
    input_dim=input_dim,
    num_loops=7
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=CONFIG['lr']
)


len_trajectory = CONFIG['len_trajectory']
ppo_batch_size = CONFIG['ppo_batch_size']

print_info('Start training ... ')
logs = CONFIG['logs']
neptune_logs = init_neptune(['simple_ppo', 'sparse_reward', '10_matmul'] + [k+':'+str(v) for (k, v) in CONFIG.items()]) if logs else None

if logs:
    neptune_logs["config_files"].upload_files([
        './utils/*.py',
        './simple_ppo.py'
    ])


# print(list(model.parall_fc.parameters())[0])

tqdm_range = tqdm(range(CONFIG['steps']), desc='Main loop')
for step in tqdm_range:

    trajectory = collect_trajectory(CONFIG['len_trajectory'], model, logs=logs)

    loss = ppo_update(
        trajectory, 
        model, 
        optimizer, 
        ppo_epochs=CONFIG['ppo_epochs'], 
        ppo_batch_size=CONFIG['ppo_batch_size'], 
        logs=logs
    )

    torch.save(model.state_dict(), 'models/ppo_model_1.pt')

    print('\n\n\n\nloss', loss, '\n\n\n\n')
    

if logs:
    neptune_logs.stop()
    
print_info('End training ... ')
print_success('YOUPI')

