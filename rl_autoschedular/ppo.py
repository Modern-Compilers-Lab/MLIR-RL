import torch
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.trajectory import TrajectoryCollector, TrajectoryData
from rl_autoschedular.observation import Observation, NumLoops
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular import config as cfg
from rl_autoschedular import file_logger as fl
from rl_autoschedular import device
from utils.log import print_error
from tqdm import trange


def collect_trajectory(model: Model, env: Env, step: int):
    """Collect a trajectory using the model and the environment.

    Args:
        model (MyModel): The model to use.
        env (Env): The environment to use.
        step (int): The current step of the main loop

    Returns:
        TrejectoryData: The collected trajectory.
    """
    tc = TrajectoryCollector()

    eps = None
    if 'epsilon' in cfg.exploration:
        ratio = step / cfg.nb_iterations
        final_eps = 0.001
        eps = final_eps + (cfg.init_epsilon - final_eps) * (1 - ratio)

    for _ in trange(cfg.bench_count, desc='Trajectory'):
        state = env.reset()
        bench_done = False
        while not bench_done:
            obs = Observation.from_state(state)
            action_index, action_bev_log_p, entropy = model.sample(obs.to(device), eps=eps)
            assert action_index.size(0) == 1 and action_bev_log_p.size(0) == 1
            action = ActionSpace.action_by_index(action_index[0], state)

            next_state, reward, op_done, speedup = env.step(state, action)
            next_obs = Observation.from_state(next_state)

            if op_done:
                next_state, bench_done = env.get_next_op_state(next_state)

            tc.append((
                Observation.get_part(obs, NumLoops).long().item(),
                action_index,
                obs,
                next_obs,
                action_bev_log_p.item(),
                reward,
                bench_done
            ))

            fl['train/entropy'].append(entropy.item())
            fl['train/reward'].append(reward)
            if speedup is not None:
                fl['train/speedup'].append(speedup)
                fl[f'train/{state.operation_features.operation_type.value}_speedup'].append(speedup)

            state = next_state
        fl['train/final_speedup'].append(speedup)

    return tc.to_trajectory()


def ppo_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer):
    """Update the model using PPO.

    Args:
        trajectory (TrajectoryData): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.

    Returns:
        float: The average loss.
    """
    trajectory.update_attributes(model)

    ppo_trange = trange(cfg.ppo_epochs, desc='PPO Epochs')
    for _ in ppo_trange:
        for batch in trajectory.loader(cfg.ppo_batch_size, 2 * cfg.truncate * cfg.bench_count):
            batch: list[torch.Tensor] = [e.to(device, non_blocking=True) for e in batch]
            (
                _,
                actions_index,
                obs,
                _,
                actions_bev_log_p,
                _, _,
                values,
                _,
                actions_old_log_p,
                off_policy_rates,
                returns,
                advantages,
            ) = batch
            max_abs_adv = advantages.abs().max()
            if cfg.normalize_adv == 'standard' and advantages.size(0) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif cfg.normalize_adv == 'max-abs' and max_abs_adv > 0:
                advantages = advantages / max_abs_adv

            with torch.enable_grad():
                actions_log_p, new_values, entropy = model(obs, actions_index)

                policy_loss, clip_frac = model.policy_model.loss(actions_log_p, actions_bev_log_p, off_policy_rates, advantages)
                loss = policy_loss

                if cfg.value_epochs == 0:
                    value_loss = model.value_model.loss(new_values, values, returns)
                    loss += cfg.value_coef * value_loss

                if 'entropy' in cfg.exploration:
                    entropy_loss = -entropy.mean()
                    loss += cfg.entropy_coef * entropy_loss

            approx_kl = (actions_old_log_p - actions_log_p).pow(2).mean() / 2

            optimizer.zero_grad()
            try:
                loss.backward()
                clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(f'Error during PPO update: {e}')

            # Logging
            ppo_trange.set_postfix({
                'loss': loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item() if cfg.value_epochs == 0 else None
            })
            fl['train_ppo/policy_loss'].append(policy_loss.item())
            fl['train_ppo/clip_frac'].append(clip_frac.item())
            fl['train_ppo/clip_factor'].append(clip_factor.item())
            fl['train_ppo/approx_kl'].append(approx_kl.item())
            if cfg.value_epochs == 0:
                fl['train_ppo/value_loss'].append(value_loss.item())
            if 'entropy' in cfg.exploration:
                fl['train_ppo/entropy_loss'].append(entropy_loss.item())


def value_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer):
    """Update the value model using the trajectory.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
    """
    trajectory.update_attributes(model)

    value_trange = trange(cfg.value_epochs, desc='Value Epochs')
    for _ in value_trange:
        for batch in trajectory.loader(cfg.value_batch_size, 2 * cfg.truncate * cfg.bench_count):
            batch: list[torch.Tensor] = [e.to(device, non_blocking=True) for e in batch]
            (
                _, _,
                obs,
                _, _, _, _,
                values,
                _, _, _,
                returns,
                _,
            ) = batch
            with torch.enable_grad():
                new_values = model.value_model(obs)

                loss = model.value_model.loss(new_values, values, returns)

            optimizer.zero_grad()
            try:
                loss.backward()
                clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(f'Error during Value update: {e}')

            # Logging
            value_trange.set_postfix({'loss': loss.item()})
            fl['train_value/loss'].append(loss.item())
            fl['train_value/clip_factor'].append(clip_factor.item())


def evaluate_benchmark(model: Model, env: Env):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (Env): The environment to use.
    """
    speedup_values: list[float] = []
    eval_trange = trange(len(env.benchmarks_data), desc='Evaluation')
    for i in eval_trange:
        # Reset the environement with the specific benchmark
        state = env.reset(i)
        bench_done = False
        cumulative_reward = 0
        while not bench_done:
            obs = Observation.from_state(state)
            action_index, _, entropy = model.sample(obs, greedy=True)
            assert action_index.size(0) == 1
            action = ActionSpace.action_by_index(action_index[0], state)

            next_state, reward, op_done, speedup = env.step(state, action)

            fl['eval/entropy'].append(entropy.item())
            if op_done:
                cumulative_reward += reward
                fl['eval/reward'].append(reward)
            if speedup is not None:
                fl[f'eval/{state.operation_features.operation_type.value}_speedup'].append(speedup)

            if op_done:
                bench_name = state.bench_name
                exec_time = next_state.exec_time
                transformation_history = next_state.transformation_history.copy()

                next_state, bench_done = env.get_next_op_state(next_state)
                if bench_done:
                    print(f"\033[92m\n- Bench: {bench_name}\n- Schedule:\n{transformation_history}\n- Speedup: {speedup}\033[0m")
                    fl[f'eval/exec_time/{bench_name}'].append(exec_time)
                    fl[f'eval/speedup/{bench_name}'].append(speedup)

            state = next_state

        fl['eval/cumulative_reward'].append(cumulative_reward)
        assert speedup is not None
        fl['eval/final_speedup'].append(speedup)
        speedup_values.append(speedup)

    if len(speedup_values) > 0:
        fl['eval/average_speedup'].append(sum(speedup_values) / len(speedup_values))
