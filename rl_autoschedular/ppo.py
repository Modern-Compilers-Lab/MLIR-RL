import torch
from rl_autoschedular.env import Env
from rl_autoschedular.model import HiearchyModel as Model
from rl_autoschedular.trajectory import TrajectoryCollector, TrajectoryData
from rl_autoschedular import config as cfg
from rl_autoschedular import file_logger as fl
from utils.log import print_error
from tqdm import trange


def collect_trajectory(model: Model, env: Env, step: int, device: torch.device = torch.device('cpu')):
    """Collect a trajectory using the model and the environment.

    Args:
        model (MyModel): The model to use.
        env (Env): The environment to use.
        neptune_logs (neptune.Run): The neptune run to log to if any. Defaults to None.
        device (torch.device): The device to use. Defaults to torch.device('cpu').

    Returns:
        TrejectoryData: The collected trajectory.
    """
    tc = TrajectoryCollector()

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
            action, action_index, action_bev_log_p, _, entropy = model.sample(obs, torch.tensor([num_loops]), eps=eps)
            assert len(action) == 1 and action_index.size(0) == 1 and action_bev_log_p.size(0) == 1

            next_state, next_obs, reward, op_done, speedup = env.step(state, action[0])

            if 'curiosity' in cfg.exploration:
                next_state_latent, next_state_latent_hat, _ = model.icm_model(obs, next_obs, action_index)
                intrinsic_reward = cfg.reward_scale * model.icm_model.forward_model.loss(next_state_latent, next_state_latent_hat).item()
                reward = (1 - cfg.intrinsic_reward_integration) * reward + cfg.intrinsic_reward_integration * intrinsic_reward

            if op_done:
                next_state, tmp_next_obs, bench_done = env.get_next_op_state(next_state)

            tc.append((
                num_loops,
                action_index,
                obs,
                next_obs,
                action_bev_log_p.item(),
                reward,
                bench_done
            ))

            if op_done:
                next_obs = tmp_next_obs

            fl['train/entropy'].append(entropy.item())
            fl['train/reward'].append(reward)
            if 'curiosity' in cfg.exploration:
                fl['train/intrinsic_reward'].append(intrinsic_reward)
            if speedup is not None:
                fl['train/speedup'].append(speedup)
                fl[f'train/{state.operation_features.operation_type}_speedup'].append(speedup)

            state = next_state
            obs = next_obs
        fl['train/final_speedup'].append(speedup)

    return tc.to_trajectory()


def ppo_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer, device: torch.device = torch.device('cpu')):
    """Update the model using PPO.

    Args:
        trajectory (TrajectoryData): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').

    Returns:
        float: The average loss.
    """
    trajectory.update_attributes(model)

    ppo_trange = trange(cfg.ppo_epochs, desc='PPO Epochs')
    for _ in ppo_trange:

        for (
            num_loops,
            actions_index,
            obs,
            next_obs,
            actions_bev_log_p,
            _, _,
            values,
            _,
            actions_old_log_p,
            off_policy_rates,
            returns,
            advantages,
        ) in trajectory.loader(cfg.ppo_batch_size):
            # Advantages batch normalization
            max_abs_adv = advantages.abs().max()
            if cfg.normalize_adv == 'standard':
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            elif cfg.normalize_adv == 'max-abs' and max_abs_adv > 0:
                advantages = advantages / max_abs_adv

            with torch.enable_grad():
                actions_log_p, new_values, entropy = model(obs, num_loops, actions_index)

                policy_loss = model.policy_model.loss(actions_log_p, actions_bev_log_p, off_policy_rates, advantages)
                loss = policy_loss

                if cfg.value_epochs == 0:
                    value_loss = model.value_model.loss(new_values, values, returns)
                    loss += cfg.value_coef * value_loss

                if 'curiosity' in cfg.exploration:
                    next_states_latent, next_states_latent_hat, action_logits = model.icm_model(obs, next_obs, actions_index)
                    curiosity_loss = model.icm_model.loss(next_states_latent, next_states_latent_hat, action_logits, actions_index)
                    loss += cfg.curiosity_coef * curiosity_loss
                if 'entropy' in cfg.exploration:
                    entropy_loss = -entropy.mean()
                    loss += cfg.entropy_coef * entropy_loss

            # clip_frac = (torch.abs((ratios - 1.0)) > 0.2).float().mean()
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
            # log_clip_frac.append(clip_frac.item())
            fl['train_ppo/clip_factor'].append(clip_factor.item())
            fl['train_ppo/approx_kl'].append(approx_kl.item())
            if cfg.value_epochs == 0:
                fl['train_ppo/value_loss'].append(value_loss.item())
            if 'curiosity' in cfg.exploration:
                fl['train_ppo/curiosity_loss'].append(curiosity_loss.item())
            if 'entropy' in cfg.exploration:
                fl['train_ppo/entropy_loss'].append(entropy_loss.item())


def value_update(trajectory: TrajectoryData, model: Model, optimizer: torch.optim.Optimizer, device: torch.device = torch.device('cpu')):
    """Update the value model using the trajectory.

    Args:
        trajectory (Trajectory): The trajectory to use.
        model (Model): The model to update.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
    """
    trajectory.update_attributes(model)

    value_trange = trange(cfg.value_epochs, desc='Value Epochs')
    for _ in value_trange:

        for (
            _, _,
            obs,
            _, _, _, _,
            values,
            _, _, _,
            returns,
            _,
        ) in trajectory.loader(cfg.value_batch_size):
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


def evaluate_benchmark(model: Model, env: Env, device: torch.device = torch.device('cpu')):
    """Evaluate the benchmark using the model.

    Args:
        model (Model): The model to use.
        env (Env): The environment to use.
        neptune_logs (Optional[neptune.Run]): The neptune run to log to if any. Defaults to None.
        device (torch.device): The device to use. Defaults to torch.device('cpu').
    """
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
            action, _, _, _, entropy = model.sample(obs, torch.tensor([len(state.operation_features.nested_loops)]), greedy=True)
            assert len(action) == 1

            # Apply the action and get the next state
            next_state, next_obs, reward, op_done, speedup = env.step(state, action[0])

            fl['eval/entropy'].append(entropy.item())
            if op_done:
                cumulative_reward += reward
                fl['eval/reward'].append(reward)
            if speedup is not None:
                fl[f'eval/{state.operation_features.operation_type}_speedup'].append(speedup)

            if op_done:
                bench_name = state.bench_name
                exec_time = next_state.exec_time
                transformation_history = next_state.transformation_history.copy()

                next_state, next_obs, bench_done = env.get_next_op_state(next_state)
                if bench_done:
                    optimizations_lines = []
                    last_idx = 0
                    for op in transformation_history:
                        if op[0] == 'done':
                            optimizations_lines.insert(last_idx, f"\t- operation {op[1][0]}:")
                            last_idx = len(optimizations_lines)
                        elif op[0] in ['no_transformation', 'vectorization']:
                            optimizations_lines.append(f"\t\t- {op[0].replace('_', ' ')}()")
                        elif op[0] == 'parallelization':
                            optimizations_lines.append(f"\t\t- tiled parallelization({', '.join(map(str, op[1]))})")
                        else:
                            optimizations_lines.append(f"\t\t- {op[0]}({', '.join(map(str, op[1]))})")
                    optimizations_lines_str = '\n'.join(optimizations_lines)
                    print(f"\033[92m\n- Bench: {bench_name}\n- Schedule:\n{optimizations_lines_str}\n- Speedup: {speedup}\033[0m")
                    fl[f'eval/exec_time/{bench_name}'].append(exec_time)
                    fl[f'eval/speedup/{bench_name}'].append(speedup)

            state = next_state
            obs = next_obs

        fl['eval/cumulative_reward'].append(cumulative_reward)
        assert speedup is not None
        fl['eval/final_speedup'].append(speedup)
        speedup_values.append(speedup)

    if len(speedup_values) > 0:
        fl['eval/average_speedup'].append(sum(speedup_values) / len(speedup_values))
