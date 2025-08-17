import torch
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.state import OperationState
from rl_autoschedular.trajectory import TrajectoryCollector, TrajectoryData
from rl_autoschedular.observation import Observation, NumLoops
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.benchmarks import Benchmarks
from rl_autoschedular.evaluation import get_code_cache_key, update_execution_cache
from rl_autoschedular import config as cfg, device
from utils.file_logger import FileLogger
from utils.log import print_error, print_info, print_success
from utils.dask_manager import DaskManager
from tqdm import trange
from time import time
import os
from typing import Optional

T_collection = tuple[list[list[float]], list[float], list[Optional[int]], int, int]


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

    print_info(f"Trajectory collection using {dm.num_workers} workers...", end=' ')
    traj_start = time()

    # Prepare benchmarks to explore
    indices = torch.randperm(len(data))[:cfg.bench_count].long().tolist()
    if len(indices) < cfg.bench_count:
        indices = (indices * cfg.bench_count)[:cfg.bench_count]
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

    traj_end_sampling = time()
    sampling_time_ms = int((traj_end_sampling - traj_start) * 1000)

    futures = []
    per_worker = cfg.bench_count // dm.num_workers
    rest = cfg.bench_count % dm.num_workers
    start = 0
    for i, worker in enumerate(dm.workers_names):
        end = start + per_worker
        if i < rest:
            end += 1
        future = dm.client.submit(__execute_states, dm.remote_train_data, states[start:end], tmp_exec_data_file, workers=[worker])
        futures.append(future)
        start = end
    results: list[T_collection] = dm.client.gather(futures)
    all_rewards = []
    all_speedups = []
    all_exec_times = []
    all_cache_misses = 0
    max_worker_time = 0
    for r, s, e, m, t in results:
        all_rewards += r
        all_speedups += s
        all_exec_times += e
        all_cache_misses += m
        if t > max_worker_time:
            max_worker_time = t
    cache_miss_rate = all_cache_misses / cfg.bench_count * 100
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
    update_execution_cache(new_cache_data, tmp_exec_data_file)

    traj_end = time()
    exec_time_ms = int((traj_end - traj_end_sampling) * 1000)
    comm_overhead = exec_time_ms - max_worker_time
    time_ms = int((traj_end - traj_start) * 1000)
    print(f"{time_ms}ms, sampling: {sampling_time_ms}ms, exec: {exec_time_ms}ms, max worker: {max_worker_time}ms, overhead: {comm_overhead}ms, cache miss rate: {cache_miss_rate:.2f}%")

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
    data_loader = trajectory.loader(cfg.ppo_batch_size, 1)

    ppo_trange = trange(cfg.ppo_epochs, desc='PPO Epochs')
    for _ in ppo_trange:
        for batch in data_loader:
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
    data_loader = trajectory.loader(cfg.value_batch_size, 1)

    value_trange = trange(cfg.value_epochs, desc='Value Epochs')
    for _ in value_trange:
        for batch in data_loader:
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
    eval_start = time()

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

    futures = []
    per_worker = len(data) // dm.num_workers
    rest = len(data) % dm.num_workers
    start = 0
    for i, worker in enumerate(dm.workers_names):
        end = start + per_worker
        if i < rest:
            end += 1
        future = dm.client.submit(__execute_states, dm.remote_eval_data, states[start:end], tmp_exec_data_file, workers=[worker])
        futures.append(future)
        start = end
    results: list[T_collection] = dm.client.gather(futures)
    all_rewards = []
    all_speedups = []
    all_exec_times = []
    for r, s, e, _, _ in results:
        all_rewards += r
        all_speedups += s
        all_exec_times += e
    new_cache_data: dict[str, dict[str, int]] = {}
    for state, rewards, speedup, exec_time in zip(states, all_rewards, all_speedups, all_exec_times):
        fl['eval/reward'].extend(rewards)
        fl['eval/cumulative_reward'].append(sum(rewards))
        fl['eval/final_speedup'].append(speedup)
        if exec_time is not None:
            fl[f'eval/exec_time/{state.bench_name}'].append(exec_time)
            fl[f'eval/speedup/{state.bench_name}'].append(speedup)
            cache_key = get_code_cache_key(state, data[state.bench_idx])
            if state.bench_name not in new_cache_data:
                new_cache_data[state.bench_name] = {}
            new_cache_data[state.bench_name][cache_key] = exec_time

        print_success("Bench:", state.bench_name)
        print_info(state.transformation_history)

    if len(all_speedups) > 0:
        fl['eval/average_speedup'].append(sum(all_speedups) / len(all_speedups))
    update_execution_cache(new_cache_data, tmp_exec_data_file)

    eval_end = time()
    time_ms = int((eval_end - eval_start) * 1000)
    print_info(f"Evaluation time: {time_ms}ms")


def __execute_states(benchs: Benchmarks, states: list[OperationState], tmp_exec_data_file: str) -> T_collection:
    worker_start = time()

    all_rewards: list[list[float]] = []
    all_speedups: list[float] = []
    all_exec_times: list[Optional[int]] = []
    cache_misses = 0
    env = Env()

    for state in states:
        env.reset(benchs, state.bench_idx)
        rewards, speedup, new_exec_time, cache_miss = env.apply_sequence(state, tmp_exec_data_file)
        all_rewards.append(rewards)
        all_speedups.append(speedup)
        all_exec_times.append(new_exec_time)
        cache_misses += int(cache_miss)

    os.remove(env.tmp_file)
    worker_end = time()
    worker_time_ms = int((worker_end - worker_start) * 1000)

    return all_rewards, all_speedups, all_exec_times, cache_misses, worker_time_ms
