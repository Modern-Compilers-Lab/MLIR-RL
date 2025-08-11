import torch
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.trajectory import TrajectoryCollector, TrajectoryData
from rl_autoschedular.observation import Observation, NumLoops
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.benchmarks import Benchmarks
from rl_autoschedular import config as cfg, device
from utils.file_logger import FileLogger
from utils.log import print_error
from utils.dask_manager import DaskManager
from dask.distributed import print
from typing import Optional
from tqdm import trange
from time import time

T_collection = tuple[TrajectoryCollector, list[float], list[float], list[float], dict[str, list[float]]]
T_evaluation = tuple[list[float], list[float], list[float], list[float], dict[str, list[float]], dict[str, tuple[int, float]]]


def collect_trajectory(data_size: int, step: int):
    """Collect a trajectory using the model and the environment.

    Args:
        model (MyModel): The model to use.
        env (Env): The environment to use.
        step (int): The current step of the main loop

    Returns:
        TrejectoryData: The collected trajectory.
    """
    dm = DaskManager()
    tc = TrajectoryCollector()
    fl = FileLogger()

    eps = None
    if 'epsilon' in cfg.exploration:
        ratio = step / cfg.nb_iterations
        final_eps = 0.001
        eps = final_eps + (cfg.init_epsilon - final_eps) * (1 - ratio)

    print(f"Trajectory collection using {dm.num_workers} workers...", end=' ')
    start = time()
    indices = torch.randperm(data_size)[:cfg.bench_count].long().tolist()
    params = [(dm.remote_train_data, indices[i:cfg.bench_count:dm.num_workers], fl.last_model_path, eps) for i in range(dm.num_workers)]
    futures = dm.client.map(__collect_benchs, *zip(*params))
    results: list[T_collection] = dm.client.gather(futures)
    end = time()
    time_ms = int((end - start) * 1000)
    print(f"{time_ms}ms")

    tc = sum([r[0] for r in results], tc)
    for _, entropies, rewards, speedups, ops_speedups in results:
        fl['train/entropy'].extend(entropies)
        fl['train/reward'].extend(rewards)
        fl['train/speedup'].extend(speedups)
        fl['train/final_speedup'].append(speedups[-1])
        for op_type, op_speedups in ops_speedups.items():
            fl[f'train/{op_type}_speedup'].extend(op_speedups)

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


def evaluate_benchmarks(data_size: int):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (Env): The environment to use.
    """
    dm = DaskManager()
    fl = FileLogger()

    print("Evaluation started...")
    start = time()
    params = [(dm.remote_eval_data, list(range(i, data_size, dm.num_workers)), fl.last_model_path) for i in range(dm.num_workers)]
    futures = dm.client.map(__evaluate_benchs, *zip(*params))
    results: list[T_evaluation] = dm.client.gather(futures)
    end = time()
    time_ms = int((end - start) * 1000)
    print(f"Evaluation time: {time_ms}ms")

    all_speedups = []
    for entropies, rewards, cumulative_rewards, final_speedups, ops_speedups, benchs_stats in results:
        fl['eval/entropy'].extend(entropies)
        fl['eval/reward'].extend(rewards)
        fl['eval/cumulative_reward'].extend(cumulative_rewards)
        fl['eval/final_speedup'].extend(final_speedups)
        all_speedups.extend(final_speedups)
        for op_type, op_speedups in ops_speedups.items():
            fl[f'eval/{op_type}_speedup'].extend(op_speedups)
        for bench_name, (exec_time, speedup) in benchs_stats.items():
            fl[f'eval/exec_time/{bench_name}'].append(exec_time)
            fl[f'eval/speedup/{bench_name}'].append(speedup)
    if len(all_speedups) > 0:
        fl['eval/average_speedup'].append(sum(all_speedups) / len(all_speedups))


def __collect_benchs(benchs: Benchmarks, indices: list[int], last_model_path: str, eps: Optional[float]) -> T_collection:
    entropies = []
    rewards = []
    speedups = []
    ops_speedups: dict[str, list[float]] = {}
    tc = TrajectoryCollector()
    env = Env(is_training=False)
    model = Model()
    model.load_state_dict(torch.load(last_model_path, weights_only=True))

    for idx in indices:
        state = env.reset(benchs, idx)
        bench_done = False
        while not bench_done:
            obs = Observation.from_state(state)
            action_index, action_bev_log_p, entropy = model.sample(obs, eps=eps)
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

            entropies.append(entropy.item())
            rewards.append(reward)
            if speedup is not None:
                speedups.append(speedup)
                op_type = state.operation_features.operation_type.value
                if op_type not in ops_speedups:
                    ops_speedups[op_type] = []
                ops_speedups[op_type].append(speedup)

            state = next_state

    return tc, entropies, rewards, speedups, ops_speedups


def __evaluate_benchs(benchs: Benchmarks, indices: list[int], last_model_path: str) -> T_evaluation:
    entropies = []
    rewards = []
    cumulative_rewards = []
    final_speedups = []
    ops_speedups: dict[str, list[float]] = {}
    benchs_stats: dict[str, tuple[int, float]] = {}
    env = Env()
    model = Model()
    model.load_state_dict(torch.load(last_model_path, weights_only=True))

    for idx in indices:
        state = env.reset(benchs, idx)
        bench_done = False
        cumulative_reward = 0
        while not bench_done:
            obs = Observation.from_state(state)
            action_index, _, entropy = model.sample(obs, greedy=True)
            assert action_index.size(0) == 1
            action = ActionSpace.action_by_index(action_index[0], state)

            next_state, reward, op_done, speedup = env.step(state, action)

            entropies.append(entropy.item())
            if op_done:
                cumulative_reward += reward
                rewards.append(reward)
            if speedup is not None:
                op_type = state.operation_features.operation_type.value
                if op_type not in ops_speedups:
                    ops_speedups[op_type] = []
                ops_speedups[op_type].append(speedup)

            if op_done:
                bench_name = state.bench_name
                exec_time = next_state.exec_time
                transformation_history = next_state.transformation_history.copy()

                next_state, bench_done = env.get_next_op_state(next_state)
                if bench_done:
                    print(f"- Bench: {bench_name}\n- Schedule:\n{transformation_history}\n- Speedup: {speedup}")
                    benchs_stats[bench_name] = (exec_time, speedup)

            state = next_state

        cumulative_rewards.append(cumulative_reward)
        assert speedup is not None
        final_speedups.append(speedup)

    return entropies, rewards, cumulative_rewards, final_speedups, ops_speedups, benchs_stats
