import numpy as np
import torch
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.trajectory import TrajectoryCollector, TrajectoryData
from rl_autoschedular.observation import Observation, NumLoops
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular import config as cfg
from rl_autoschedular import file_logger as fl , offline_data_collector
from rl_autoschedular import device
from utils.log import print_error
from tqdm import trange
import time


def collect_trajectory(model: Model, env: Env, step: int):
    """Collect a trajectory using the model and the environment (on-policy for PPO).

    Args:
        model (Model): The policy/value model.
        env (Env): The environment.
        step (int): Current training step.

    Returns:
        TrajectoryData: The collected trajectory.
    """
    tc = TrajectoryCollector()
    
    env_time = 0.0  # Time spent in environment steps

    eps = None
    if 'epsilon' in cfg.exploration:  # optional exploration schedule
        ratio = step / cfg.nb_iterations
        final_eps = 0.001
        eps = final_eps + (cfg.init_epsilon - final_eps) * (1 - ratio)
    
    # store rewards and entropies to log average for the model accross the benchmarks later    
    all_speedups = []
    all_entropies = []

    for _ in trange(cfg.bench_count, desc='Trajectory'):
        
        t0 = time.perf_counter()
        state = env.reset()
        env_time += time.perf_counter() - t0
        bench_done = False
        speedup = None
        
        # store rewards and entropies to log average for the current benchmark later
        bench_rewards, bench_entropies = [], []

        bench_name = state.bench_name


        while not bench_done:
            obs = Observation.from_state(state)

            # Sample action and log-prob from *current policy*
            action_index, action_log_p, entropy = model.sample(obs.to(device), eps=eps)    
            
            assert action_index.size(0) == 1 and action_log_p.size(0) == 1
            action = ActionSpace.action_by_index(action_index[0], state)

            # Step environment
            t0 = time.perf_counter()
            next_state, reward, op_done, speedup = env.step(state, action)
            env_time += time.perf_counter() - t0
            next_obs = Observation.from_state(next_state)
            

            if op_done:
                t0 = time.perf_counter()
                next_state, bench_done = env.get_next_op_state(next_state)
                env_time += time.perf_counter() - t0


            tc.append((
                Observation.get_part(obs, NumLoops).long().item(),
                action_index,
                obs,
                next_obs,
                reward,
                bench_done,
            ))
            
            if cfg.collect_offline_data:
                offline_data_collector.add_transition(
                    obs,
                    action_index,
                    next_obs,
                    reward,
                    bench_done
                )
            

            # Accumulate metrics
            bench_rewards.append(reward)
            bench_entropies.append(entropy.item())
            state = next_state
            
         # === Per-benchmark logging ===
        mean_reward = float(np.mean(bench_rewards)) if bench_rewards else 0.0
        mean_entropy = float(np.mean(bench_entropies)) if bench_entropies else 0.0
        
        all_speedups.append(speedup)
        all_entropies.extend(bench_entropies)
        
        
        bench_metrics = {
            "mean_reward": mean_reward,
            "mean_entropy": mean_entropy,
            "final_speedup": speedup if speedup is not None else 0.0,
        }
        
        fl.log_scalars(f"train/{bench_name}", bench_metrics, step)

     # === Global logging (across all benchmarks) ===
    if all_speedups:
        fl.log_scalar("train/average_speedup", float(np.mean(all_speedups)), step)
    if all_entropies:
        fl.log_scalar("train/average_entropy", float(np.mean(all_entropies)), step)
        
    if cfg.collect_offline_data:
        offline_data_collector.flush()

    return tc.to_trajectory() , env_time


def ppo_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer,step):
    """Update the model using PPO (on-policy).

    Args:
        trajectory (TrajectoryData): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.

    Returns:
        float: The average loss across updates.
    """
    trajectory.update_attributes(model)

    avg_loss = 0.0
    total_steps = 0
    
    
    metrics_accum = {
        "policy_loss": 0.0,
        "clip_frac": 0.0,
        "clip_factor": 0.0,
        "approx_kl": 0.0,
        "value_loss": 0.0,
        "mean_entropy": 0.0,
    }
    metrics_count = 0

    ppo_trange = trange(cfg.ppo_epochs, desc='PPO Epochs')
    for _ in ppo_trange:
        for batch in trajectory.loader(cfg.ppo_batch_size, shuffle=True):
            
            batch = [e.to(device, non_blocking=True) for e in batch]
            (
                _,
                actions_index,
                obs,
                _,
                _,
                _,
                values,
                _,
                actions_old_log_p,
                returns,
                advantages,
            ) = batch
            
            # Normalize advantages if configured
            max_abs_adv = advantages.abs().max()
            if cfg.normalize_adv == 'standard' and advantages.size(0) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif cfg.normalize_adv == 'max-abs' and max_abs_adv > 0:
                advantages = advantages / max_abs_adv

            with torch.enable_grad():
                # Forward pass through policy and value
                actions_log_p, new_values, entropy = model(obs, actions_index)

                # PPO policy loss (clipped surrogate)
                policy_loss, clip_frac = model.policy_model.loss(
                    actions_log_p, actions_old_log_p, advantages
                )
                loss = policy_loss

                # Value loss
                if cfg.value_epochs == 0:  # update value jointly
                    value_loss = model.value_model.loss(new_values, values, returns)
                    loss += cfg.value_coef * value_loss

                # Entropy bonus
                if 'entropy' in cfg.exploration:
                    entropy_loss = -entropy.mean()
                    loss += cfg.entropy_coef * entropy_loss

            # KL estimate
            approx_kl = (actions_old_log_p - actions_log_p).pow(2).mean() / 2

            # Gradient step
            optimizer.zero_grad()
            try:
                loss.backward()
                clip_factor = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()
            except Exception as e:
                print_error(f'Error during PPO update: {e}')
                continue

            # Logging
            avg_loss += loss.item() * advantages.size(0)
            total_steps += advantages.size(0)

            ppo_trange.set_postfix({
                'loss': loss.item(),
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item() if cfg.value_epochs == 0 else None
            })
            
            # Accumulate weighted averages
            
            bs = advantages.size(0)
            avg_loss += loss.item() * bs
            total_steps += bs

            metrics_accum["policy_loss"] += policy_loss.item() * bs
            metrics_accum["clip_frac"]   += clip_frac.item() * bs
            metrics_accum["clip_factor"] += clip_factor.item() * bs
            metrics_accum["approx_kl"]   += approx_kl.item() * bs
            if cfg.value_epochs == 0:
                metrics_accum["value_loss"] += value_loss.item() * bs
            if 'entropy' in cfg.exploration:
                metrics_accum["mean_entropy"] -= entropy_loss.item() * bs

            metrics_count += bs
            
    # final averaging
    final_metrics = {k: (v / metrics_count) for k, v in metrics_accum.items() if v != 0}
    fl.log_scalars("PPO_Training", final_metrics, step)




def value_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer,step):
    """Update the value model using the trajectory.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
    """
    trajectory.update_attributes(model)
    
    metrics_accum = {
        "loss": 0.0,
        "clip_factor": 0.0,
    }
    total_steps = 0


    value_trange = trange(cfg.value_epochs, desc='Value Epochs')
    for _ in value_trange:
        for batch in trajectory.loader(cfg.value_batch_size, shuffle=True):
            
            batch = [e.to(device, non_blocking=True) for e in batch]
            (
                _, _,
                obs,
                _, _,      # next_obs, rewards
                 _,      # done
                values,
                _, _,      # next_values, actions_old_log_p
                returns,
                _,         # advantages
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

            # Accumulate weighted averages
            bs = obs.size(0)
            metrics_accum["loss"] += loss.item() * bs
            metrics_accum["clip_factor"] += clip_factor.item() * bs
            total_steps += bs
    # === Final averaging ===
    if total_steps > 0:
        final_metrics = {k: v / total_steps for k, v in metrics_accum.items()}
        fl.log_scalars("Value_Training", final_metrics, step)
            
@torch.no_grad()       
def evaluate_benchmarks(model: Model, env: Env, step: int):
    """Evaluta a the model on the evaluation environment.

    Args:
        model (Model): The policy/value model.
        env (Env): The environment.
        step (int): Current training step.

    Returns:
        env_time (float): Time spent in environment steps.
    """

    
    env_time = 0.0  # Time spent in environment steps

    eps = None

    
    # store rewards and entropies to log average for the model accross the benchmarks later    
    all_speedups = []
    all_entropies = []


    for _ in trange(cfg.bench_count, desc='Trajectory'):
        
        t0 = time.perf_counter()
        state = env.reset()
        env_time += time.perf_counter() - t0
        bench_done = False
        speedup = None
        
        # store rewards and entropies to log average for the current benchmark later
        bench_rewards, bench_entropies = [], []

        bench_name = state.bench_name


        while not bench_done:
            obs = Observation.from_state(state)

            # Sample action and log-prob from *current policy*
            action_index, action_log_p, entropy = model.sample(obs.to(device), eps=eps)
            assert action_index.size(0) == 1 and action_log_p.size(0) == 1
            action = ActionSpace.action_by_index(action_index[0], state)

            # Step environment
            t0 = time.perf_counter()
            next_state, reward, op_done, speedup = env.step(state, action)
            env_time += time.perf_counter() - t0
            next_obs = Observation.from_state(next_state)
            

            if op_done:
                t0 = time.perf_counter()
                next_state, bench_done = env.get_next_op_state(next_state)
                env_time += time.perf_counter() - t0

            
            # Accumulate metrics
            bench_rewards.append(reward)
            bench_entropies.append(entropy.item())
            state = next_state
            
         # === Per-benchmark logging ===
        mean_reward = float(np.mean(bench_rewards)) if bench_rewards else 0.0
        mean_entropy = float(np.mean(bench_entropies)) if bench_entropies else 0.0
        
        all_speedups.append(speedup)
        all_entropies.extend(bench_entropies)
        
        
        bench_metrics = {
            "mean_reward": mean_reward,
            "mean_entropy": mean_entropy,
            "final_speedup": speedup if speedup is not None else 0.0,
        }
        
        fl.log_scalars(f"eval/{bench_name}", bench_metrics, step)
        
        print(
            f"\033[92m\n- Eval Bench: {bench_name}\n"
            f"- Mean Reward: {mean_reward:.4f}\n"
            f"- Mean Entropy: {mean_entropy:.4f}\n"
            f"- Final Speedup: {speedup if speedup is not None else 0.0:.4f}\033[0m"
        )


     # === Global logging (across all benchmarks) ===
    if all_speedups:
        fl.log_scalar("eval/average_speedup", float(np.mean(all_speedups)), step)
    if all_entropies:
        fl.log_scalar("eval/average_entropy", float(np.mean(all_entropies)), step)

    return  env_time