import torch
from typing import Optional, Union
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.state import OperationState
from rl_autoschedular import config as cfg
from rl_autoschedular import file_logger as fl
from dataclasses import dataclass
from utils.log import print_error, print_success
from tqdm import trange


@dataclass
class Trajectory:
    """Dataclass to store the trajectory data."""
    states: list[OperationState]
    """States in the trajectory."""
    actions: list[tuple[str, Optional[Union[list[int], int]]]]
    """Actions in the trajectory."""
    obs: torch.Tensor
    """Observations in the trajectory"""
    values: torch.Tensor
    """Values of actions in the trajectory."""
    next_obs: torch.Tensor
    """Observations of next states in the trajectory."""
    next_values: torch.Tensor
    """Values of actions in the trajectory with one additional step (shifted to one step in the future)."""
    action_log_p: torch.Tensor
    """Action log probabilities in the trajectory."""
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
            obs=torch.concatenate((self.obs, other.obs)),
            values=torch.concatenate((self.values, other.values)),
            next_obs=torch.concatenate((self.next_obs, other.next_obs)),
            next_values=torch.concatenate((self.next_values, other.next_values)),
            action_log_p=torch.concatenate((self.action_log_p, other.action_log_p)),
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
            obs=self.obs.clone(),
            values=self.values.clone(),
            next_obs=self.next_obs.clone(),
            next_values=self.next_values.clone(),
            action_log_p=self.action_log_p.clone(),
            rewards=self.rewards.clone(),
            done=self.done.clone(),
        )


def collect_trajectory(model: Model, env: Env, step: int, device: torch.device = torch.device('cpu')):
    """Collect a trajectory using the model and the environment.

    Args:
        model (MyModel): The model to use.
        env (Env): The environment to use.
        neptune_logs (neptune.Run): The neptune run to log to if any. Defaults to None.
        device (torch.device): The device to use. Defaults to torch.device('cpu').

    Returns:
        Trajectory: The collected trajectory.
    """
    stored_state: list[OperationState] = []
    stored_action_index: list[tuple[str, Optional[Union[list[int], int]]]] = []
    stored_obs: list[torch.Tensor] = []
    stored_value: list[torch.Tensor] = []
    stored_next_obs: list[torch.Tensor] = []
    stored_next_value: list[torch.Tensor] = []
    stored_action_log_p: list[torch.Tensor] = []
    stored_reward: list[torch.Tensor] = []
    stored_done: list[torch.Tensor] = []

    log_rewards: list[float] = []
    log_intrinsic_rewards: list[float] = []
    log_speedups: list[float] = []
    log_entropy: list[float] = []
    log_final_speedups: list[float] = []
    log_op_speedups: dict[str, list[float]] = {}

    eps = None
    if 'epsilon' in cfg.exploration:
        ratio = step / cfg.nb_iterations
        final_eps = 0.001
        eps = final_eps + (cfg.init_epsilon - final_eps) * (1 - ratio)

    traj_trange = trange(cfg.bench_count, desc='Trajectory')
    for _ in traj_trange:
        state, obs = env.reset()
        traj_trange.set_postfix({'eps': eps, 'bench': state.bench_name})
        bench_done = False
        while not bench_done:
            num_loops = len(state.operation_features.nested_loops)
            action_index, action_log_p, value, entropy = model.sample(obs, [num_loops], eps=eps)

            assert len(action_index) == 1
            next_state, next_obs, reward, op_done, speedup = env.step(state, action_index[0])
            next_value = model.value_model(next_obs)

            if 'curiosity' in cfg.exploration:
                next_state_latent, next_state_latent_hat, _ = model.icm_model(obs, next_obs, action_index)
                intrinsic_reward = cfg.reward_scale * model.icm_model.forward_model.loss(next_state_latent, next_state_latent_hat).item()
                reward = (1 - cfg.intrinsic_reward_integration) * reward + cfg.intrinsic_reward_integration * intrinsic_reward

            stored_state.append(state)
            stored_action_index.append(action_index[0])
            stored_obs.append(obs)
            stored_value.append(value)
            stored_next_obs.append(next_obs)
            stored_next_value.append(next_value)
            stored_action_log_p.append(action_log_p)
            stored_reward.append(torch.tensor(reward).unsqueeze(0))

            if op_done:
                next_state, next_obs, bench_done = env.get_next_op_state(next_state)

            stored_done.append(torch.tensor(bench_done).unsqueeze(0))

            log_entropy.append(entropy.item())
            log_rewards.append(reward)
            if 'curiosity' in cfg.exploration:
                log_intrinsic_rewards.append(intrinsic_reward)
            if speedup is not None:
                log_speedups.append(speedup)
                if state.operation_features.operation_type not in log_op_speedups:
                    log_op_speedups[state.operation_features.operation_type] = []
                log_op_speedups[state.operation_features.operation_type].append(speedup)

            state = next_state
            obs = next_obs
        log_final_speedups.append(speedup)

    fl['train/entropy'].extend(log_entropy)
    fl['train/reward'].extend(log_rewards)
    fl['train/speedup'].extend(log_speedups)
    fl['train/final_speedup'].extend(log_final_speedups)
    for op_type, speedups in log_op_speedups.items():
        fl[f'train/{op_type}_speedup'].extend(speedups)
    if 'curiosity' in cfg.exploration:
        fl['train/intrinsic_reward'].extend(log_intrinsic_rewards)

    stored_obs_tensor = torch.concatenate(stored_obs)
    stored_value_tensor = torch.concatenate(stored_value)
    stored_next_obs_tensor = torch.concatenate(stored_next_obs)
    stored_next_value_tensor = torch.concatenate(stored_next_value)
    stored_action_log_p_tensor = torch.concatenate(stored_action_log_p)
    stored_reward_tensor = torch.concatenate(stored_reward).float()
    stored_done_tensor = torch.concatenate(stored_done).float()

    trajectory = Trajectory(
        states=stored_state,
        actions=stored_action_index,
        obs=stored_obs_tensor,
        values=stored_value_tensor,
        next_obs=stored_next_obs_tensor,
        next_values=stored_next_value_tensor,
        action_log_p=stored_action_log_p_tensor,
        rewards=stored_reward_tensor,
        done=stored_done_tensor,
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


def shuffle_ppo_data(stored_action_index: list[tuple[str, Optional[Union[list[int], int]]]], stored_states: list[OperationState], *tensors: torch.Tensor):
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

    permutation = torch.randperm(len(stored_action_index))

    stored_action_index = [stored_action_index[i] for i in permutation]
    stored_states = [stored_states[i] for i in permutation]

    return stored_action_index, stored_states, *shuffle_tensors(*tensors, permutation=permutation)


def shuffle_tensors(*tensors: torch.Tensor, permutation: Optional[torch.Tensor] = None):
    """Shuffle the tensors.

    Args:
        *tensors (torch.Tensor): The tensors to shuffle.

    Returns:
        tuple[torch.Tensor]: The shuffled tensors.
    """
    permutation = torch.randperm(tensors[0].shape[0]) if permutation is None else permutation
    return tuple(tensor[permutation] for tensor in tensors)


def compute_gae(done: torch.Tensor, rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor, gamma: float = 1, lambda_: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Generalized Advantage Estimation.

    Args:
        done (torch.Tensor): done flags.
        rewards (torch.Tensor): rewards.
        values (torch.Tensor): values.
        next_values (torch.Tensor): values of the next state.
        gamma (float): discount factor. Defaults to 1.
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
    returns = advantages + values

    return advantages, returns


def compute_returns(done: torch.Tensor, rewards: torch.Tensor, gamma: float = 1) -> torch.Tensor:
    """Compute the returns.

    Args:
        done (torch.Tensor): done flags.
        rewards (torch.Tensor): rewards.
        gamma (float): discount factor. Defaults to 1.

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


def ppo_update(trajectory: Trajectory, model: Model, optimizer: torch.optim.Optimizer, device: torch.device = torch.device('cpu')):
    """Update the model using PPO.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').

    Returns:
        float: The average loss.
    """

    log_policy_loss: list[float] = []
    log_entropy_loss: list[float] = []
    log_value_loss: list[float] = []
    log_curiosity_loss: list[float] = []
    # log_clip_frac: list[float] = []
    log_approx_kl: list[float] = []

    ppo_trange = trange(cfg.ppo_epochs, desc='PPO Epochs')
    for _ in ppo_trange:
        stored_states = trajectory.states
        stored_action_index = trajectory.actions
        stored_obs = trajectory.obs
        stored_values = trajectory.values
        stored_next_obs = trajectory.next_obs
        stored_next_values = trajectory.next_values
        stored_action_log_p = trajectory.action_log_p
        stored_rewards = trajectory.rewards
        stored_done = trajectory.done

        stored_values = stored_values.reshape(-1)
        stored_next_values = stored_next_values.reshape(-1)
        stored_rewards = stored_rewards.reshape(-1)
        stored_done = stored_done.reshape(-1)

        stored_advantages, _ = compute_gae(stored_done, stored_rewards, stored_values, stored_next_values)
        stored_returns = compute_returns(stored_done, stored_rewards)

        stored_action_index, stored_states, stored_obs, stored_values, stored_next_obs, stored_action_log_p, stored_advantages, stored_returns = shuffle_ppo_data(
            stored_action_index,
            stored_states,
            stored_obs,
            stored_values,
            stored_next_obs,
            stored_action_log_p,
            stored_advantages,
            stored_returns
        )

        len_trajectory = len(stored_action_index)
        for i in range(0, len_trajectory, cfg.ppo_batch_size):
            betch_end = min(i + cfg.ppo_batch_size, len_trajectory)

            actions = stored_action_index[i:betch_end]
            states = stored_states[i:betch_end]
            obs = stored_obs[i:betch_end]
            values = stored_values[i:betch_end]
            next_obs = stored_next_obs[i:betch_end]
            actions_log_p = stored_action_log_p[i:betch_end]
            advantages = stored_advantages[i:betch_end]
            returns = stored_returns[i:betch_end]

            if cfg.normalize_adv and advantages.shape[0] > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            with torch.enable_grad():
                new_actions_log_p, new_values, entropy = model(obs, [len(state.operation_features.nested_loops) for state in states], actions=actions)

                policy_loss = model.policy_model.loss(new_actions_log_p, actions_log_p, advantages)
                loss = policy_loss

                if cfg.value_epochs == 0:
                    value_loss = model.value_model.loss(new_values, values, returns)
                    loss += cfg.value_coef * value_loss

                if 'curiosity' in cfg.exploration:
                    next_states_latent, next_states_latent_hat, action_logits = model.icm_model(obs, next_obs, actions)
                    curiosity_loss = model.icm_model.loss(next_states_latent, next_states_latent_hat, action_logits, actions)
                    loss += cfg.curiosity_coef * curiosity_loss
                if 'entropy' in cfg.exploration:
                    entropy_loss = -entropy.mean()
                    loss += cfg.entropy_coef * entropy_loss

            # clip_frac = (torch.abs((ratios - 1.0)) > 0.2).float().mean()
            approx_kl = (actions_log_p - new_actions_log_p).pow(2).mean() / 2

            optimizer.zero_grad()
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(f'Error during PPO update: {e}')

            # Logging
            ppo_trange.set_postfix({
                'loss': loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item() if cfg.value_epochs == 0 else None,
                'approx_kl': approx_kl.item()
            })
            log_policy_loss.append(policy_loss.item())
            # log_clip_frac.append(clip_frac.item())
            log_approx_kl.append(approx_kl.item())
            if cfg.value_epochs == 0:
                log_value_loss.append(value_loss.item())
            if 'curiosity' in cfg.exploration:
                log_curiosity_loss.append(curiosity_loss.item())
            if 'entropy' in cfg.exploration:
                log_entropy_loss.append(entropy_loss.item())

    fl['train_ppo/policy_loss'].extend(log_policy_loss)
    # fl['train_ppo/clip_factor'].extend(log_clip_frac)
    fl['train_ppo/approx_kl'].extend(log_approx_kl)
    if cfg.value_epochs == 0:
        fl['train_ppo/value_loss'].extend(log_value_loss)
    if 'curiosity' in cfg.exploration:
        fl['train_ppo/curiosity_loss'].extend(log_curiosity_loss)
    if 'entropy' in cfg.exploration:
        fl['train_ppo/entropy_loss'].extend(log_entropy_loss)


def value_update(trajectory: Trajectory, model: Model, optimizer: torch.optim.Optimizer, device: torch.device = torch.device('cpu')):
    """Update the value model using the trajectory.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
    """
    log_loss: list[float] = []

    value_trange = trange(cfg.value_epochs, desc='Value Epochs')
    for _ in value_trange:
        stored_obs = trajectory.obs
        stored_values = trajectory.values
        stored_rewards = trajectory.rewards
        stored_done = trajectory.done

        stored_values = stored_values.reshape(-1)
        stored_rewards = stored_rewards.reshape(-1)
        stored_done = stored_done.reshape(-1)

        stored_returns = compute_returns(stored_done, stored_rewards)

        stored_obs, stored_values, stored_returns = shuffle_tensors(stored_obs, stored_values, stored_returns)

        len_trajectory = stored_values.shape[0]
        for i in range(0, len_trajectory, cfg.value_batch_size):
            betch_end = min(i + cfg.value_batch_size, len_trajectory)

            obs = stored_obs[i:betch_end]
            values = stored_values[i:betch_end]
            returns = stored_returns[i:betch_end]

            with torch.enable_grad():
                new_values = model.value_model(obs)

                loss = model.value_model.loss(new_values, values, returns)

            optimizer.zero_grad()
            try:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(f'Error during Value update: {e}')

            # Logging
            value_trange.set_postfix({'loss': loss.item()})
            log_loss.append(loss.item())

    fl['train_value/loss'].extend(log_loss)


def update_trajectory_values(trajectory: Trajectory, model: Model, device: torch.device = torch.device('cpu')):
    """Update the values in the trajectory after fitting the value model.

    Args:
        trajectory (Trajectory): The trajectory to update.
        model (Model): The model to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
    """
    stored_obs = trajectory.obs
    stored_next_obs = trajectory.next_obs

    # Update the values in the trajectory
    stored_values = model.value_model(stored_obs)
    stored_next_values = model.value_model(stored_next_obs)

    # Update the trajectory
    trajectory.values = stored_values
    trajectory.next_values = stored_next_values


def evaluate_benchmark(model: Model, env: Env, device: torch.device = torch.device('cpu')):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (Env): The environment to use.
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
    """

    log_entropy: list[float] = []
    log_reward: list[float] = []
    log_cumulative_reward: list[float] = []
    log_final_speedup: list[float] = []
    log_op_speedups: dict[str, list[float]] = {}

    speedup_values: list[float] = []
    nbr_benchmarks = len(env.benchmarks_data)
    eval_trange = trange(nbr_benchmarks, desc='Evaluation')
    for i in eval_trange:
        # Reset the environement with the specific benchmark
        state, obs = env.reset(i)

        bench_done = False
        cumulative_reward = 0
        while not bench_done:
            # Select the action using the model
            action, _, _, entropy = model.sample(obs, [len(state.operation_features.nested_loops)], greedy=True)

            # Apply the action and get the next state
            assert len(action) == 1
            next_state, next_obs, reward, op_done, speedup = env.step(state, action[0])

            log_entropy.append(entropy.item())
            if op_done:
                cumulative_reward += reward
                log_reward.append(reward)
            if speedup is not None:
                if state.operation_features.operation_type not in log_op_speedups:
                    log_op_speedups[state.operation_features.operation_type] = []
                log_op_speedups[state.operation_features.operation_type].append(speedup)

            if op_done:
                bench_name = state.bench_name
                exec_time = next_state.exec_time
                transformation_history = next_state.transformation_history.copy()
                next_state, next_obs, bench_done = env.get_next_op_state(next_state)
                if bench_done:
                    eval_trange.set_postfix({'bench': bench_name, 'speedup': speedup})
                    print_success(f"Bench: {bench_name}, Schedule: {transformation_history} - {speedup}")
                    fl[f'eval/exec_time/{bench_name}'].append(exec_time)
                    fl[f'eval/speedup/{bench_name}'].append(speedup)

            state = next_state
            obs = next_obs

        log_cumulative_reward.append(cumulative_reward)
        assert speedup is not None
        log_final_speedup.append(speedup)
        speedup_values.append(speedup)

    if len(speedup_values) > 0:
        fl['eval/average_speedup'].append(sum(speedup_values) / len(speedup_values))

    fl['eval/entropy'].extend(log_entropy)
    fl['eval/reward'].extend(log_reward)
    fl['eval/cumulative_reward'].extend(log_cumulative_reward)
    fl['eval/final_speedup'].extend(log_final_speedup)
    for op_type, speedups in log_op_speedups.items():
        fl[f'eval/{op_type}_speedup'].extend(speedups)
