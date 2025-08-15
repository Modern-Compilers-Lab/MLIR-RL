import torch
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.state import OperationState
from rl_autoschedular.trajectory import TrajectoryCollector, TrajectoryData
from rl_autoschedular.observation import Observation, NumLoops
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.benchmarks import Benchmarks
from rl_autoschedular.evaluation import get_code_cache_key, bulk_update_execution_cache
from rl_autoschedular import config as cfg, device
from utils.file_logger import FileLogger
from utils.log import print_error, print_info
from utils.dask_manager import DaskManager
from tqdm import trange
from time import time
import os
from typing import Optional

T_collection = tuple[list[list[float]], list[float], list[Optional[int]]]
T_evaluation = tuple[list[list[float]], list[float], list[float], dict[str, list[float]], dict[str, tuple[OperationState, int, float]]]


def collect_trajectory(data: Benchmarks, model: Model, step: int, tmp_exec_data_file: str):
    """Collect a trajectory using the model and the environment.

    Args:
        model (MyModel): The model to use.
        env (Env): The environment to use.
        step (int): The current step of the main loop
        tmp_exec_data_file (str): The path to the temporary execution data file.

    Returns:
        TrejectoryData: The collected trajectory.
    """
    dm = DaskManager()
    fl = FileLogger()

    eps = None
    if 'epsilon' in cfg.exploration:
        ratio = step / cfg.nb_iterations
        final_eps = 0.001
        eps = final_eps + (cfg.init_epsilon - final_eps) * (1 - ratio)

    print_info(f"Trajectory collection using {dm.num_workers} workers...")
    start = time()

    # Prepare benchmarks to explore
    indices = torch.randperm(len(data))[:cfg.bench_count].long().tolist()
    envs: list[Env] = []
    states: list[OperationState] = []
    observations: list[torch.Tensor] = []
    tcs: list[TrajectoryCollector] = []
    for idx in indices:
        env = Env(create_tmp_file=False)
        state = env.reset(data, idx)
        envs.append(env)
        states.append(state)
        observations.append(Observation.from_state(state))
        tcs.append(TrajectoryCollector())

    while (active_states := [(i, s) for i, s in enumerate(states) if not s.terminal]):
        # Sample states that are not terminal yet
        obss = torch.cat([observations[i] for i, _ in active_states])
        actions_index, actions_bev_log_p, entropies = model.sample(obss.to(device), eps=eps)
        fl['train/entropy'].extend(entropies.tolist())

        # Record data and update states
        for (i, state), obs, action_index, action_bev_log_p in zip(active_states, obss, actions_index, actions_bev_log_p):
            obs = obs.unsqueeze(0)

            # Get action and use it to get next state
            action = ActionSpace.action_by_index(action_index, state)
            states[i] = next_state = envs[i].step(state, action)
            observations[i] = next_obs = Observation.from_state(next_state)

            # If the benchmark is not done yet, keep next operation state instead
            done = False
            if next_state.terminal:
                next_op_state = envs[i].get_next_op_state(next_state)
                if next_op_state is not None:
                    states[i] = next_op_state
                    observations[i] = Observation.from_state(next_op_state)
                else:
                    done = True

            # Record available data
            tcs[i].append((
                Observation.get_part(obs, NumLoops).long().item(),
                action_index.unsqueeze(0),
                obs,
                next_obs,
                action_bev_log_p.item(),
                0.0,  # This will be filled after execution
                done
            ))

    params = [(dm.remote_train_data, states[i:i + dm.num_workers], tmp_exec_data_file) for i in range(0, cfg.bench_count, dm.num_workers)]
    futures = dm.client.map(__collect_benchs, *zip(*params))
    results: list[T_collection] = dm.client.gather(futures)
    all_rewards = sum([r for r, _, _ in results], [])
    all_speedups = sum([s for _, s, _ in results], [])
    all_exec_times = sum([e for _, _, e in results], [])
    new_cache_data: dict[str, dict[str, int]] = {}
    for tc, state, rewards, speedup, exec_time in zip(tcs, states, all_rewards, all_speedups, all_exec_times):
        # Update trajectory
        tc.rewards = rewards
        # Log metrics
        fl['train/reward'].extend(rewards)
        fl['train/final_speedup'].append(speedup)
        # Get new cache data
        if exec_time is not None:
            cache_key = get_code_cache_key(state, data[state.bench_idx])
            if state.bench_name not in new_cache_data:
                new_cache_data[state.bench_name] = {}
            new_cache_data[state.bench_name][cache_key] = exec_time

    tc = sum(tcs, TrajectoryCollector())
    bulk_update_execution_cache(new_cache_data, tmp_exec_data_file)

    end = time()
    time_ms = int((end - start) * 1000)
    print_info(f"{time_ms}ms")

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
    fl = FileLogger()
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
                actions_log_p, new_values, entropies = model(obs, actions_index)

                policy_loss, clip_frac = model.policy_model.loss(actions_log_p, actions_bev_log_p, off_policy_rates, advantages)
                loss = policy_loss

                if cfg.value_epochs == 0:
                    value_loss = model.value_model.loss(new_values, values, returns)
                    loss += cfg.value_coef * value_loss

                if 'entropy' in cfg.exploration:
                    entropy_loss = -entropies.mean()
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
    fl = FileLogger()
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


def evaluate_benchmarks(model: Model, data: Benchmarks, tmp_exec_data_file: str):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (Env): The environment to use.
        tmp_exec_data_file (str): The path to the temporary execution data file.
    """
    dm = DaskManager()
    fl = FileLogger()

    print_info("Evaluation started...")
    start = time()

    # Prepare benchmarks to explore
    indices = range(len(data))
    envs: list[Env] = []
    states: list[OperationState] = []
    observations: list[torch.Tensor] = []
    for idx in indices:
        env = Env(create_tmp_file=False)
        state = env.reset(data, idx)
        envs.append(env)
        states.append(state)
        observations.append(Observation.from_state(state))

    while (active_states := [(i, s) for i, s in enumerate(states) if not s.terminal]):
        # Sample states that are not terminal yet
        obss = torch.cat([observations[i] for i, _ in active_states])
        actions_index, _, entropies = model.sample(obss.to(device), greedy=True)
        fl['eval/entropy'].extend(entropies.tolist())

        # Record data and update states
        for (i, state), obs, action_index in zip(active_states, obss, actions_index):
            obs = obs.unsqueeze(0)

            # Get action and use it to get next state
            action = ActionSpace.action_by_index(action_index, state)
            states[i] = next_state = envs[i].step(state, action)
            observations[i] = Observation.from_state(next_state)

            # If the benchmark is not done yet, keep next operation state instead
            if next_state.terminal:
                next_op_state = envs[i].get_next_op_state(next_state)
                if next_op_state is not None:
                    states[i] = next_op_state
                    observations[i] = Observation.from_state(next_op_state)

    params = [(dm.remote_train_data, states[i:i + dm.num_workers], tmp_exec_data_file) for i in range(0, cfg.bench_count, dm.num_workers)]
    futures = dm.client.map(__evaluate_benchs, *zip(*params))
    results: list[T_evaluation] = dm.client.gather(futures)
    end = time()
    time_ms = int((end - start) * 1000)
    print_info(f"Evaluation time: {time_ms}ms")

    all_speedups: list[float] = []
    new_cache_data: dict[str, dict[str, int]] = {}
    for all_rewards, cumulative_rewards, final_speedups, ops_speedups, benchs_stats in results:
        fl['eval/reward'].extend(sum(all_rewards, []))
        fl['eval/cumulative_reward'].extend(cumulative_rewards)
        fl['eval/final_speedup'].extend(final_speedups)
        all_speedups.extend(final_speedups)
        for op_type, op_speedups in ops_speedups.items():
            fl[f'eval/{op_type}_speedup'].extend(op_speedups)
        for bench_name, (state, exec_time, speedup) in benchs_stats.items():
            fl[f'eval/exec_time/{bench_name}'].append(exec_time)
            fl[f'eval/speedup/{bench_name}'].append(speedup)
            cache_key = get_code_cache_key(state, data[state.bench_idx])
            if bench_name not in new_cache_data:
                new_cache_data[bench_name] = {}
            new_cache_data[bench_name][cache_key] = exec_time
    if len(all_speedups) > 0:
        fl['eval/average_speedup'].append(sum(all_speedups) / len(all_speedups))
    bulk_update_execution_cache(new_cache_data, tmp_exec_data_file)


def __collect_benchs(benchs: Benchmarks, states: list[OperationState], tmp_exec_data_file: str) -> T_collection:
    all_rewards: list[list[float]] = []
    all_speedups: list[float] = []
    all_exec_times: list[Optional[int]] = []
    env = Env()

    for state in states:
        env.reset(benchs, state.bench_idx)
        rewards, speedup, new_exec_time = env.apply_sequence(state, tmp_exec_data_file)
        all_rewards.append(rewards)
        all_speedups.append(speedup)
        all_exec_times.append(new_exec_time)

    os.remove(env.tmp_file)

    return all_rewards, all_speedups, all_exec_times


def __evaluate_benchs(benchs: Benchmarks, states: list[OperationState], tmp_exec_data_file: str) -> T_evaluation:
    all_rewards: list[list[float]] = []
    cumulative_rewards: list[float] = []
    all_speedups: list[float] = []
    ops_speedups: dict[str, list[float]] = {}
    benchs_stats: dict[str, tuple[int, float]] = {}

    env = Env()

    for state in states:
        env.reset(benchs, state.bench_idx)
        rewards, speedup, new_exec_time = env.apply_sequence(state, tmp_exec_data_file)
        all_rewards.append(rewards)
        cumulative_rewards.append(sum(rewards))
        all_speedups.append(speedup)
        op_type = state.operation_features.operation_type.value
        if op_type not in ops_speedups:
            ops_speedups[op_type] = []
        ops_speedups[op_type].append(speedup)
        if new_exec_time is not None:
            benchs_stats[state.bench_name] = (state, new_exec_time, speedup)

    os.remove(env.tmp_file)

    return all_rewards, cumulative_rewards, all_speedups, ops_speedups, benchs_stats
