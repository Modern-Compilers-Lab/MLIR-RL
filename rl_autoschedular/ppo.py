import torch
import neptune
from typing import Optional, Union
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.state import OperationState
from rl_autoschedular import config as cfg
from dataclasses import dataclass
from utils.log import print_info, print_success
from tqdm import trange


@dataclass
class Trajectory:
    """Dataclass to store the trajectory data."""
    states: list[OperationState]
    """States in the trajectory."""
    actions: list[tuple[str, Optional[Union[list[int], int]]]]
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

    def extend(self, other: 'Trajectory') -> 'Trajectory':
        """Extend the trajectory with another trajectory.

        Args:
            other (Trajectory): The other trajectory to extend with.

        Returns:
            Trajectory: The extended trajectory.
        """
        return Trajectory(
            states=self.states + [state.copy() for state in other.states],
            actions=self.actions + other.actions,
            values=torch.concatenate((self.values, other.values)),
            next_values=torch.concatenate((self.next_values, other.next_values)),
            action_log_p=torch.concatenate((self.action_log_p, other.action_log_p)),
            x=torch.concatenate((self.x, other.x)),
            rewards=torch.concatenate((self.rewards, other.rewards)),
            done=torch.concatenate((self.done, other.done)),
        )

    def copy(self) -> 'Trajectory':
        """Copy the trajectory.

        Returns:
            Trajectory: The copied trajectory.
        """
        return Trajectory(
            states=[state.copy() for state in self.states],
            actions=self.actions.copy(),
            values=self.values.clone(),
            next_values=self.next_values.clone(),
            action_log_p=self.action_log_p.clone(),
            x=self.x.clone(),
            rewards=self.rewards.clone(),
            done=self.done.clone(),
        )


def collect_trajectory(model: Model, env: Env, device: torch.device = torch.device('cpu'), neptune_logs: Optional[neptune.Run] = None):
    """Collect a trajectory using the model and the environment.

    Args:
        model (MyModel): The model to use.
        env (Env): The environment to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.

    Returns:
        Trajectory: The collected trajectory.
    """

    batch_state, batch_obs = env.reset()
    batch_obs = batch_obs.to(device)

    stored_state: list[OperationState] = []
    stored_action_index: list[tuple[str, Optional[Union[list[int], int]]]] = []
    stored_value: list[torch.Tensor] = []
    stored_action_log_p: list[torch.Tensor] = []
    stored_x: list[torch.Tensor] = []
    stored_reward: list[torch.Tensor] = []
    stored_done: list[torch.Tensor] = []

    traj_trange = trange(cfg.bench_count, desc='Trajectory')
    for i in traj_trange:
        traj_trange.set_postfix({'bench': batch_state.bench_name})
        bench_done = False
        while not bench_done:
            num_loops = len(batch_state.operation_features.nested_loops)
            action_index, action_log_p, values, entropy = model.sample(batch_obs, [num_loops])

            assert len(action_index) == 1
            batch_next_state, batch_next_obs, batch_reward, batch_terminated, batch_speedup, bench_done = env.step(batch_state, action_index[0])

            stored_state.append(batch_state)
            stored_action_index.append(action_index[0])
            stored_value.append(values)
            stored_action_log_p.append(action_log_p)
            stored_x.append(batch_obs)
            stored_reward.append(torch.tensor(batch_reward).unsqueeze(0))
            stored_done.append(torch.tensor(batch_terminated).unsqueeze(0))

            if neptune_logs is not None:
                neptune_logs['train/entropy'].append(entropy.item())
                if batch_terminated:
                    neptune_logs['train/reward'].append(batch_reward)
                if batch_speedup is not None:
                    neptune_logs['train/speedup'].append(batch_speedup)
                    neptune_logs[f'train/{batch_state.operation_type}_speedup'].append(batch_speedup)

            batch_state = batch_next_state
            batch_obs = batch_next_obs
        if neptune_logs is not None:
            neptune_logs['train/final_speedup'].append(batch_speedup)

    next_value = model.sample_value(batch_obs)

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


def shuffle_ppo_data(stored_action_index: list[tuple[str, Optional[Union[list[int], int]]]], stored_state: list[OperationState], stored_action_log_p: torch.Tensor, stored_x: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor):
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

    permutation = torch.randperm(stored_action_log_p.shape[0])

    stored_action_index = [stored_action_index[i] for i in permutation]
    stored_state = [stored_state[i] for i in permutation]
    stored_action_log_p = stored_action_log_p[permutation]
    stored_x = stored_x[permutation]
    advantages = advantages[permutation]
    returns = returns[permutation]

    return stored_action_index, stored_state, stored_action_log_p, stored_x, advantages, returns


def shuffle_value_data(stored_x: torch.Tensor, stored_returns: torch.Tensor):
    """Shuffle the value data.

    Args:
        stored_x (torch.Tensor): stored observation vectors.
        stored_returns (torch.Tensor): stored returns.

    Returns:
        torch.Tensor: shuffled observation vectors.
        torch.Tensor: shuffled returns.
    """

    permutation = torch.randperm(stored_x.shape[0])

    stored_x = stored_x[permutation]
    stored_returns = stored_returns[permutation]

    return stored_x, stored_returns


def compute_gae(done: torch.Tensor, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
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
    last_advantage = 0

    for t in reversed(range(done.shape[0])):
        mask = 1.0 - done[t]
        last_value = next_values[t] * mask
        last_advantage = last_advantage * mask

        delta = rewards[t] + gamma * last_value - values[t]
        last_advantage = delta + gamma * lambda_ * last_advantage

        advantages[t] = last_advantage

    return advantages


def compute_returns(done: torch.Tensor, rewards: torch.Tensor, gamma: float = 0.99) -> torch.Tensor:
    """Compute the returns.

    Args:
        done (torch.Tensor): done flags.
        rewards (torch.Tensor): rewards.
        values (torch.Tensor): values.
        gamma (float): discount factor. Defaults to 0.99.

    Returns:
        torch.Tensor: returns.
    """
    assert len(rewards) == len(done)

    returns = torch.zeros(done.shape[0], dtype=torch.float32)
    last_return = 0

    for t in reversed(range(done.shape[0])):
        mask = 1.0 - done[t]
        last_return = last_return * mask

        last_return = rewards[t] + gamma * last_return

        returns[t] = last_return

    return returns


def ppo_update(trajectory: Trajectory, model: Model, optimizer: torch.optim.Optimizer, device: torch.device = torch.device('cpu'), neptune_logs: Optional[neptune.Run] = None):
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

    ppo_trange = trange(cfg.ppo_epochs, desc='PPO Epochs')
    for _ in ppo_trange:
        stored_state = trajectory.states
        stored_action_index = trajectory.actions
        stored_value = trajectory.values
        stored_next_value = trajectory.next_values
        stored_action_log_p = trajectory.action_log_p
        stored_x = trajectory.x
        stored_reward = trajectory.rewards
        stored_done = trajectory.done

        stored_value = stored_value.reshape(-1)
        stored_next_value = stored_next_value.reshape(-1)
        stored_reward = stored_reward.reshape(-1)
        stored_done = stored_done.reshape(-1)

        advantages = compute_gae(stored_done, stored_reward, stored_value, stored_next_value)
        returns = compute_returns(stored_done, stored_reward)

        stored_action_index, stored_state, stored_action_log_p, stored_x, stored_advantages, stored_returns = shuffle_ppo_data(stored_action_index, stored_state, stored_action_log_p, stored_x, advantages, returns)

        len_trajectory = len(stored_action_index)
        losses = []
        for i in range(0, len_trajectory, cfg.ppo_batch_size):
            betch_end = min(i + cfg.ppo_batch_size, len_trajectory)
            actions = stored_action_index[i:betch_end]
            states = stored_state[i:betch_end]
            actions_log_p = stored_action_log_p[i:betch_end]
            advantages = stored_advantages[i:betch_end]
            returns = stored_returns[i:betch_end]
            x = stored_x[i:betch_end]

            with torch.enable_grad():
                _, new_actions_log_p, new_values, entropy = model.sample(x, [len(state.operation_features.nested_loops) for state in states], actions)

                if advantages.shape[0] > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratios = torch.exp(new_actions_log_p - actions_log_p)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
                policy_loss = - torch.min(surr1, surr2).mean()
                # policy_loss = - (new_actions_log_p * advantages).mean()

                value_loss = (returns - new_values).abs().mean()
                # value_loss = (returns - new_values).pow(2).mean()

                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses.append(loss.item())

            # Logging
            if neptune_logs is not None:
                neptune_logs['train_ppo/policy_loss'].append(policy_loss.item())
                neptune_logs['train_ppo/entropy_loss'].append(entropy.item())
                neptune_logs['train_ppo/clip_factor'].append(clip_factor.item())

        if len(losses) > 0:
            epoch_loss = sum(losses) / len(losses)
            ppo_trange.set_postfix({'loss': epoch_loss})


def value_update(trajectory: Trajectory, model: Model, optimizer: torch.optim.Optimizer, device: torch.device = torch.device('cpu'), neptune_logs: Optional[neptune.Run] = None):
    """Update the value estimation using the trajectory.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        value_epochs (int): The number of value epochs.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.

    Returns:
        float: The average loss.
    """

    value_trange = trange(cfg.value_epochs, desc='Value Epochs')
    for _ in value_trange:
        stored_x = trajectory.x
        stored_reward = trajectory.rewards
        stored_done = trajectory.done

        stored_reward = stored_reward.reshape(-1)
        stored_done = stored_done.reshape(-1)

        returns = compute_returns(stored_done, stored_reward)

        stored_x, stored_returns = shuffle_value_data(stored_x, returns)

        len_trajectory = stored_returns.shape[0]
        losses = []
        for i in range(0, len_trajectory, cfg.ppo_batch_size):
            betch_end = min(i + cfg.ppo_batch_size, len_trajectory)
            returns = stored_returns[i:betch_end]
            x = stored_x[i:betch_end]

            with torch.enable_grad():
                new_values = model.sample_value(x).reshape(-1)

                # loss = (returns - new_values).pow(2).mean()
                loss = (returns - new_values).abs().mean()

            optimizer.zero_grad()
            loss.backward()
            clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            losses.append(loss.item())

            # Logging
            if neptune_logs is not None:
                neptune_logs['train_value/value_loss'].append(loss.item())
                neptune_logs['train_value/clip_factor'].append(clip_factor.item())

        if len(losses) > 0:
            epoch_loss = sum(losses) / len(losses)
            value_trange.set_postfix({'loss': epoch_loss})


def evaluate_benchmark(model: Model, env: Env, device: torch.device = torch.device('cpu'), neptune_logs: Optional[neptune.Run] = None):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (Env): The environment to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.
    """
    speedup_values: list[float] = []
    nbr_benchmarks = len(env.benchmarks_data)
    for i in range(nbr_benchmarks):
        # Reset the environement with the specific benchmark
        state, obs = env.reset(i)
        obs = obs.to(device)
        print_info(f'Evaluation ({i}/{nbr_benchmarks}): {state.bench_name}')

        bench_done = False
        cumulative_reward = 0
        while not bench_done:
            # Select the action using the model
            action, _, _, entropy = model.sample(obs, [len(state.operation_features.nested_loops)])

            # Apply the action and get the next state
            assert len(action) == 1
            next_state, next_obs, reward, terminated, speedup, bench_done = env.step(state, action[0])

            if neptune_logs is not None:
                neptune_logs['eval/entropy'].append(entropy.item())
            if terminated:
                print_success('Reward:', reward)
                cumulative_reward += reward
                if neptune_logs is not None:
                    neptune_logs['eval/reward'].append(reward)
            if speedup is not None:
                print_success(f'Speedup: {speedup}')
                if neptune_logs is not None:
                    neptune_logs[f'eval/{state.operation_type}_speedup'].append(speedup)

            state = next_state
            obs = next_obs.to(device)

        if neptune_logs is not None:
            neptune_logs['eval/cumulative_reward'].append(cumulative_reward)
        assert speedup is not None
        print_success(f'Final Speedup: {speedup}')
        if neptune_logs is not None:
            neptune_logs['eval/final_speedup'].append(speedup)
            speedup_values.append(speedup)

    if neptune_logs is not None and len(speedup_values) > 0:
        neptune_logs['eval/average_speedup'].append(sum(speedup_values) / len(speedup_values))
