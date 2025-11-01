import dotenv

dotenv.load_dotenv()

import os
import time
import torch
import numpy as np

from rl_autoschedular import config as cfg, file_logger as fl
from utils.config import Config
from utils.file_logger import FileLogger
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.env import Env
from iql.agent import IQLAgent
from utils.data_collector import OfflineDataset
from rl_autoschedular.observation import Observation,OpFeatures, ActionHistory

from tqdm import trange

device = torch.device("cpu")

cfg = Config()
fl = FileLogger()

def load_dataset():
    """Load offline dataset from OfflineDataset singleton."""
    dataset = OfflineDataset(
        save_dir=cfg.offline_data_save_dir,
        fname=cfg.offline_data_file
    ).load()

    if not dataset:
        raise FileNotFoundError(f"Offline dataset not found: {cfg.offline_data_file}")

    states = torch.tensor(dataset["obs"], dtype=torch.float32)
    actions = torch.tensor(dataset["actions"], dtype=torch.long)
    rewards = torch.tensor(dataset["rewards"], dtype=torch.float32)
    next_states = torch.tensor(dataset["next_obs"], dtype=torch.float32)
    dones = torch.tensor(dataset["dones"], dtype=torch.float32)

    return states, actions, rewards, next_states, dones


@torch.no_grad()       
def evaluate_benchmarks(model: IQLAgent, env: Env, step: int):
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
            action_index, action_log_p, entropy = model.sample(obs.to(device))
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

def train_iql():
    # Load offline dataset
    print(f"Loading offline dataset from {cfg.offline_data_file} ...")
    states, actions, rewards, next_states, dones = load_dataset()
    dataset_size = states.shape[0]
    print(f"Dataset loaded: {dataset_size} transitions")

    # Initialize IQL agent
    agent = IQLAgent(cfg,device,obs_parts=[OpFeatures, ActionHistory])

    eval_env = Env(is_training=False,run_name=cfg.run_name)


    print("Starting IQL training ...")
    start_time = time.time()

    step = 0
    iql_trange = trange(cfg.max_steps, desc="IQL Training",dynamic_ncols=True)
    for step in iql_trange:
        # Sample a random batch
        idxs = np.random.randint(0, dataset_size, size=cfg.batch_size)
        batch = (
            states[idxs].to(device),
            actions[idxs].to(device),
            rewards[idxs].to(device),
            next_states[idxs].to(device),
            dones[idxs].to(device),
        )

        losses = agent.update(batch)


        #  Only log occasionally to reduce disk I/O
        if step % 50 == 0:
            fl.log_scalars("train", losses, step)

        if (step +1) % 100 == 0:
            elapsed = time.time() - start_time
            iql_trange.set_postfix({
                "Value Loss": f"{losses['value']:.4f}",
                "Q Loss": f"{losses['q']:.4f}",
                "Policy Loss": f"{losses['policy']:.4f}",
                "Elapsed": f"{elapsed:.2f}s"
            })

        # Evaluate the agent on benchmarks every 1000 steps
        if (step + 1) % 1000 == 0:
            print("Evaluating on benchmarks ...")
            eval_start = time.time()
            env_time = evaluate_benchmarks(agent, eval_env, step)
            eval_time = time.time() - eval_start
            print(f"Evaluation completed in {eval_time:.2f} seconds (env time: {env_time:.2f} seconds)")
            fl.flush()



        if (step+1) % 2000 == 0 and step > 0:
            ckpt_path = os.path.join(cfg.results_dir, f"iql_step_{step}.pt")
            os.makedirs(cfg.results_dir, exist_ok=True)
            torch.save(agent.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")



    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f} seconds.")


if __name__ == "__main__":
    train_iql()