import torch
import neptune
from typing import Optional
from rl_autoschedular.env import ParallelEnv
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.state import OperationState
from rl_autoschedular import config as cfg
from dataclasses import dataclass


@dataclass
class Trajectory:
    """Dataclass to store the trajectory data."""
    states: list[OperationState]
    """States in the trajectory."""
    actions: list[tuple[str, list[int]]]
    """Actions in the trajectory."""
    values: torch.Tensor
    """Values of actions in the trajectory."""
    next_values: torch.Tensor
    """Values of actions in the trajectory with one additional step (shifted to one step in the future)."""
    action_log_p: torch.Tensor
    """Action log probabilities in the trajectory."""
    x: torch.Tensor
    """Observation vectors in the trajectory."""
    rewards: torch.Tensor
    """Rewards in the trajectory."""
    done: torch.Tensor
    """Done flags in the trajectory."""


def collect_trajectory(len_trajectory: int, model: Model, env: ParallelEnv, device: torch.device = torch.device('cpu'), neptune_logs: Optional[neptune.Run] = None):
    """Collect a trajectory using the model and the environment.

    Args:
        len_trajectory (int): The length of the trajectory.
        model (MyModel): The model to use.
        env (ParallelEnv): The environment to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.

    Returns:
        Trajectory: The collected trajectory.
    """

    batch_state, batch_obs = env.reset()
    batch_obs = [obs.to(device) for obs in batch_obs]

    stored_state: list[OperationState] = []
    stored_action_index: list[tuple[str, list[int]]] = []
    stored_value: list[torch.Tensor] = []
    stored_action_log_p: list[torch.Tensor] = []
    stored_x: list[torch.Tensor] = []
    stored_reward: list[torch.Tensor] = []
    stored_done: list[torch.Tensor] = []

    # for i in tqdm(range(len_trajectory)):
    for i in range(len_trajectory):

        x = torch.cat(batch_obs)
        # with torch.no_grad():
        action_index, action_log_p, values, entropy = model.sample(x)
        new_action_index, new_action_log_p, new_values, new_entropy = model.sample(x, actions=action_index)
        assert (action_index == new_action_index), 'check the get_p yerham babak'
        assert (new_action_log_p == action_log_p).all(), 'check the get_p yerham babak'
        assert (values == new_values).all(), 'check the get_p yerham babak'
        assert (entropy == new_entropy).all(), 'check the get_p yerham babak'

        batch_next_obs, batch_reward, batch_terminated, batch_next_state, batch_final_state = env.step(batch_state, action_index)

        stored_action_index += action_index

        # NOTE: Only using one environment
        stored_state.append(batch_state[0])
        stored_value.append(values)
        stored_action_log_p.append(action_log_p)
        stored_x.append(x)
        stored_reward.append(torch.tensor(batch_reward).unsqueeze(0))
        stored_done.append(torch.tensor(batch_terminated).unsqueeze(0))

        # print(batch_next_state[0].actions)

        for i in range(env.num_env):
            done = batch_terminated[i]
            reward = batch_reward[i]
            final_state = batch_final_state[i]
            # print(done)
            if done and final_state is not None:
                speedup_metric = final_state.root_exec_time / final_state.exec_time
                print('-' * 70)
                print(f"Bench: {final_state.bench_name}")
                print(final_state.transformation_history)
                print('reward:', reward)
                print('cummulative_reward:', final_state.cummulative_reward)
                print('Speedup:', speedup_metric)
                print('Old Exec time:', final_state.root_exec_time * 10**-9, 's')
                print('New Exec time:', final_state.exec_time * 10**-9, 's')
                print('-' * 70)
                if neptune_logs is not None:
                    neptune_logs['train/final_speedup'].append(speedup_metric)
                    neptune_logs['train/cummulative_reward'].append(final_state.cummulative_reward)
                    neptune_logs[f'train/{final_state.bench_name}_speedup'].append(speedup_metric)

                # running_return_stats.add(final_state.raw_operation, speedup_metric)

        batch_state = batch_next_state
        batch_obs = batch_next_obs

    # with torch.no_grad():
    x = torch.cat(batch_obs)
    _, _, next_value, _ = model.sample(x)

    stored_value_tensor = torch.concatenate(stored_value)
    stored_action_log_p_tensor = torch.concatenate(stored_action_log_p)
    stored_x_tensor = torch.concatenate(stored_x)
    stored_reward_tensor = torch.concatenate(stored_reward).float()
    stored_done_tensor = torch.concatenate(stored_done).float()

    stored_next_value = torch.concatenate((stored_value_tensor[1:], next_value))
    assert (stored_value_tensor[1:] == stored_next_value[:-1]).all()

    trajectory = Trajectory(
        states=stored_state,
        actions=stored_action_index,
        values=stored_value_tensor.detach(),
        next_values=stored_next_value.detach(),
        action_log_p=stored_action_log_p_tensor.detach(),
        x=stored_x_tensor.detach(),
        rewards=stored_reward_tensor.detach(),
        done=stored_done_tensor.detach(),
    )

    return trajectory


def shuffle_trajectory(trajectory: Trajectory):
    """Shuffle the trajectory.

    Args:
        trajectory (Trajectory): The trajectory to shuffle.

    Returns:
        Trajectory: The shuffled trajectory.
    """

    stored_state = trajectory.states
    stored_action_index = trajectory.actions
    stored_value = trajectory.values
    stored_next_value = trajectory.next_values
    stored_action_log_p = trajectory.action_log_p
    stored_x = trajectory.x
    stored_reward = trajectory.rewards
    stored_done = trajectory.done

    permutation = torch.randperm(stored_action_log_p.size()[0])

    stored_state = [stored_state[i] for i in permutation]
    stored_action_index = [stored_action_index[i] for i in permutation]
    stored_value = stored_value[permutation]
    stored_next_value = stored_next_value[permutation]
    stored_action_log_p = stored_action_log_p[permutation]
    stored_x = stored_x[permutation]
    stored_reward = stored_reward[permutation]
    stored_done = stored_done[permutation]

    trajectory = Trajectory(
        states=stored_state,
        actions=stored_action_index,
        values=stored_value,
        next_values=stored_next_value,
        action_log_p=stored_action_log_p,
        x=stored_x,
        rewards=stored_reward,
        done=stored_done,
    )

    return trajectory


def shuffle_ppo_data(stored_action_index: list[tuple[str, list[int]]], stored_action_log_p: torch.Tensor, stored_x: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor):
    """Shuffle the PPO data.

    Args:
        stored_action_index (list[tuple[str, list[int]]]): stored action index.
        stored_action_log_p (torch.Tensor): stored action log probabilities.
        stored_x (torch.Tensor): stored observation vectors.
        advantages (torch.Tensor): stored advantages.
        returns (torch.Tensor): stored returns.

    Returns:
        list[tuple[str, list[int]]]: shuffled action index.
        torch.Tensor: shuffled action log probabilities.
        torch.Tensor: shuffled observation vectors.
        torch.Tensor: shuffled advantages.
        torch.Tensor: shuffled returns.
    """

    permutation = torch.randperm(stored_action_log_p.size()[0])

    stored_action_index = [stored_action_index[i] for i in permutation]
    stored_action_log_p = stored_action_log_p[permutation]
    stored_x = stored_x[permutation]
    advantages = advantages[permutation]
    returns = returns[permutation]

    return stored_action_index, stored_action_log_p, stored_x, advantages, returns


def compute_gae(done: torch.Tensor, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, gamma: float = 0.99, lambda_: float = 0.95):
    """Compute the Generalized Advantage Estimation.

    Args:
        done (torch.Tensor): done flags.
        rewards (torch.Tensor): rewards.
        values (torch.Tensor): values.
        next_values (torch.Tensor): values of the next state.
        gamma (float): discount factor. Defaults to 0.99.
        lambda_ (float): GAE factor. Defaults to 0.95.

    Returns:
        torch.Tensor: advantages.
        torch.Tensor: returns.
    """
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


def ppo_update(trajectory: Trajectory, model: Model, optimizer: torch.optim.Optimizer, ppo_epochs: int, ppo_batch_size: int, device: torch.device = torch.device('cpu'), entropy_coef: float = 0.01, neptune_logs: Optional[neptune.Run] = None):
    """Update the model using PPO.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        ppo_epochs (int): The number of PPO epochs.
        ppo_batch_size (int): The PPO batch size.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
        entropy_coef (float): The entropy coefficient. Defaults to 0.01.
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.

    Returns:
        float: The average loss.
    """

    loss_i = 0

    for epoch in range(ppo_epochs):

        # stored_state = trajectory.states
        stored_action_index = trajectory.actions
        stored_value = trajectory.values
        stored_next_value = trajectory.next_values
        stored_action_log_p = trajectory.action_log_p
        stored_x = trajectory.x
        stored_reward = trajectory.rewards
        stored_done = trajectory.done

        len_trajectory = stored_x.shape[0]
        assert len_trajectory % ppo_batch_size == 0

        stored_value = stored_value.reshape(-1).detach()
        stored_next_value = stored_next_value.reshape(-1).detach()
        stored_reward = stored_reward.reshape(-1).detach()
        stored_done = stored_done.reshape(-1).detach()

        advantages, returns = compute_gae(stored_done, stored_reward, stored_value, stored_next_value)

        # if epoch == 0:
        #     for i in range(len(returns)):
        #         if returns[i] != 0:
        #             running_return_stats.add(stored_state[i].raw_operation, returns[i].item())

        # for i in range(len(returns)):
        #     if returns[i] != 0:
        #         returns[i] = returns[i] / running_return_stats.std(stored_state[i].raw_operation)

        stored_action_index, stored_action_log_p, stored_x, stored_advantages, stored_returns = shuffle_ppo_data(stored_action_index, stored_action_log_p, stored_x, advantages, returns)

        acc_loss = 0
        for i in range(len_trajectory // ppo_batch_size):

            begin, end = i * ppo_batch_size, (i + 1) * ppo_batch_size

            action_index = stored_action_index[begin:end]
            action_log_p = stored_action_log_p[begin:end].to(device)
            advantages = stored_advantages[begin:end].to(device)
            returns = stored_returns[begin:end].to(device)
            x = stored_x[begin:end].to(device)

            if len(action_index) == 1:
                match action_index[0][0]:
                    case 'no_transformation' | 'vectorization' | 'img2col':
                        model.interchange_fc.weight.requires_grad = False
                        model.interchange_fc.bias.requires_grad = False
                        model.tiling_fc.weight.requires_grad = False
                        model.tiling_fc.bias.requires_grad = False
                        model.parall_fc.weight.requires_grad = False
                        model.parall_fc.bias.requires_grad = False
                    case 'parallelization':
                        model.interchange_fc.weight.requires_grad = False
                        model.interchange_fc.bias.requires_grad = False
                        model.tiling_fc.weight.requires_grad = False
                        model.tiling_fc.bias.requires_grad = False
                        model.parall_fc.weight.requires_grad = True
                        model.parall_fc.bias.requires_grad = True
                    case 'tiling':
                        model.interchange_fc.weight.requires_grad = False
                        model.interchange_fc.bias.requires_grad = False
                        model.tiling_fc.weight.requires_grad = True
                        model.tiling_fc.bias.requires_grad = True
                        model.parall_fc.weight.requires_grad = False
                        model.parall_fc.bias.requires_grad = False
                    case 'interchange':
                        model.interchange_fc.weight.requires_grad = True
                        model.interchange_fc.bias.requires_grad = True
                        model.tiling_fc.weight.requires_grad = False
                        model.tiling_fc.bias.requires_grad = False
                        model.parall_fc.weight.requires_grad = False
                        model.parall_fc.bias.requires_grad = False
                    case _:
                        raise ValueError(f"Unknown action: {action_index[0][0]}")

            with torch.enable_grad():
                # New predicition:
                _, new_action_log_p, new_values, entropy = model.sample(x, actions=action_index)

                # print(advantages.round(decimals=2))

                if advantages.shape[0] > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_action_log_p, action_log_p, advantages = new_action_log_p.reshape(-1), action_log_p.reshape(-1), advantages.reshape(-1)

                ratio = torch.exp(new_action_log_p - action_log_p.detach())
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages
                policy_loss = - torch.min(surr1, surr2).mean()

                returns, new_values = returns.reshape(-1), new_values.reshape(-1)

                value_loss = ((returns - new_values)**2).mean()
                value_loss = ((returns - new_values).abs()).mean()

                loss = policy_loss - entropy_coef * entropy + 0.5 * value_loss

            optimizer.zero_grad()
            loss.backward()
            clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            acc_loss += loss.item()
            loss_i += 1

            # Collecting metircs:
            if neptune_logs is not None:
                neptune_logs['train/policy_loss'].append(policy_loss.item())
                neptune_logs['train/value_loss'].append(value_loss.item())
                neptune_logs['train/entropy'].append(entropy.item())
                neptune_logs['train/clip_factor'].append(clip_factor.item())

        # print()
        # print('***'*50)
        # print()

    return acc_loss / loss_i


def evaluate_benchmark(model: Model, env: ParallelEnv, device: torch.device = torch.device('cpu'), neptune_logs: Optional[neptune.Run] = None):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (ParallelEnv): The environment to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.
    """
    # NOTE: Only using one environment
    speedup_values: list[float] = []
    for i, (bench_name, benchmark_data) in enumerate(env.envs[0].benchmarks_data):
        if cfg.data_format == 'mlir':
            print(f'Benchmark ({i}):', bench_name)
        else:
            op_tag = benchmark_data.operation_tags[-1]
            print(f'Operation ({i}):', benchmark_data.operations[op_tag].raw_operation)

        # Reset the environement with the specific operation
        state, obs = env.reset(i)
        obs = torch.cat(obs).to(device)

        while True:

            # with torch.no_grad():
            # Select the action using the model
            action, _, _, _ = model.sample(obs)

            # Apply the action and get the next state
            next_obs, reward, terminated, next_state, final_state = env.step(state, action)

            done = terminated[0]
            final_state = final_state[0]
            if done and final_state is not None:
                speedup_metric = final_state.root_exec_time / final_state.exec_time
                print('Operation:', final_state.operation_features.raw_operation)
                print('Base execution time:', final_state.root_exec_time, 's')
                print('New execution time:', final_state.exec_time, 's')
                print('Speedup:', speedup_metric)

                if neptune_logs is not None:
                    neptune_logs[f'eval/{final_state.operation_features.raw_operation}_speedup'].append(speedup_metric)
                    neptune_logs['eval/final_speedup'].append(speedup_metric)
                    speedup_values.append(speedup_metric)

                break

            state = next_state
            obs = torch.cat(next_obs).to(device)

        print('\n\n\n')

    if neptune_logs is not None:
        neptune_logs['eval/average_speedup'].append(sum(speedup_values) / len(speedup_values))
