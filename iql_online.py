import os
import time
import torch
import numpy as np
from tqdm import trange

import dotenv
dotenv.load_dotenv()

from rl_autoschedular import config as cfg, file_logger as fl
from rl_autoschedular.env import Env
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.observation import Observation, OpFeatures, ActionHistory
from iql.iql_agent import IQLAgent
from utils.data_collector import OfflineDataset


device = torch.device("cpu")


def load_offline_dataset():
    """Load offline dataset for warm-starting replay buffer."""
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
    """Evaluate model performance across all benchmarks."""
    env_time = 0.0
    eps = None
    all_speedups, all_entropies = [], []

    for _ in trange(cfg.bench_count, desc="Eval Trajectory", leave=False):
        t0 = time.perf_counter()
        state = env.reset()
        env_time += time.perf_counter() - t0
        bench_done, speedup = False, None
        bench_rewards, bench_entropies = [], []
        bench_name = state.bench_name

        while not bench_done:
            obs = Observation.from_state(state)
            action_index, action_log_p, entropy = model.sample(obs.to(device), eps=eps)
            action = ActionSpace.action_by_index(action_index[0], state)

            t0 = time.perf_counter()
            next_state, reward, op_done, speedup = env.step(state, action)
            env_time += time.perf_counter() - t0

            if op_done:
                t0 = time.perf_counter()
                next_state, bench_done = env.get_next_op_state(next_state)
                env_time += time.perf_counter() - t0

            bench_rewards.append(reward)
            bench_entropies.append(entropy.item())
            state = next_state

        # per-benchmark logs
        fl.log_scalars(f"eval/{bench_name}", {
            "mean_reward": float(np.mean(bench_rewards)) if bench_rewards else 0.0,
            "mean_entropy": float(np.mean(bench_entropies)) if bench_entropies else 0.0,
            "final_speedup": speedup if speedup is not None else 0.0,
        }, step)

        all_speedups.append(speedup)
        all_entropies.extend(bench_entropies)

    # global logs
    if all_speedups:
        fl.log_scalar("eval/average_speedup", float(np.mean(all_speedups)), step)
    if all_entropies:
        fl.log_scalar("eval/average_entropy", float(np.mean(all_entropies)), step)

    return env_time


class ReplayBuffer:
    """Simple replay buffer mixing offline + online data."""
    def __init__(self, max_size=1000000):
        self.states, self.actions, self.rewards, self.next_states, self.dones = [], [], [], [], []
        self.max_size = max_size

    def add(self, s, a, r, ns, d):
        if len(self.states) >= self.max_size:
            # drop oldest
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
            self.dones.pop(0)

        self.states.append(s.squeeze(0))
        self.actions.append(a.squeeze(0))
        self.rewards.append(r.squeeze(0))
        self.next_states.append(ns.squeeze(0))
        self.dones.append(d)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.states), size=batch_size)
        return (
            torch.stack([self.states[i] for i in idxs]),
            torch.stack([self.actions[i] for i in idxs]),
            torch.stack([self.rewards[i] for i in idxs]),
            torch.stack([self.next_states[i] for i in idxs]),
            torch.stack([self.dones[i] for i in idxs]),
        )

    def __len__(self):
        return len(self.states)


def hybrid_finetune():
    # === Load pretrained agent ===
    agent = IQLAgent(cfg, device, obs_parts=[OpFeatures, ActionHistory])
    ckpt_path = "./iql_results/iql_step_17999.pt"
    if os.path.exists(ckpt_path):
        agent.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded pretrained checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # === Init Replay Buffer with offline data ===
    buffer = ReplayBuffer(max_size=200000)
    states, actions, rewards, next_states, dones = load_offline_dataset()
    for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
        buffer.add(s, a, r, ns, d)
    print(f"Replay buffer initialized with {len(buffer)} offline samples")

    # environments
    train_env = Env(is_training=True, run_name=cfg.run_name)
    eval_env = Env(is_training=False, run_name=cfg.run_name)

    print("Starting HYBRID fine-tuning (offline + online)...")
    start_time = time.time()
    state = train_env.reset()

    hybrid_trange = trange(cfg.max_steps, desc="Hybrid Fine-tuning", dynamic_ncols=True)
    for step in hybrid_trange:
        # reset benchmark
        state = train_env.reset()
        done = False

        while not done:
            # current obs
            obs = Observation.from_state(state)

            # agent picks action
            action_index, _, _ = agent.sample(obs.to(device), eps=None)
            action = ActionSpace.action_by_index(action_index[0], state)

            # env step
            next_state, reward, op_done, _ = train_env.step(state, action)

            # build next_obs BEFORE advancing benchmark
            next_obs = Observation.from_state(next_state)

            # if op finished, advance to next op or benchmark end
            if op_done:
                next_state, done = train_env.get_next_op_state(next_state)

            # push transition to replay buffer
            buffer.add(
                obs.to(device),
                action_index,
                torch.tensor(reward, dtype=torch.float32, device=device),
                next_obs.to(device),
                torch.tensor(done, dtype=torch.float32, device=device),
            )

            # move forward
            state = next_state

        # after benchmark, do 1 gradient update
        batch = buffer.sample(cfg.batch_size)
        losses = agent.update(batch)

        # logging
        if step % 50 == 0:
            fl.log_scalars("hybrid_train", losses, step)



        # logging
        if step % 50 == 0:
            fl.log_scalars("hybrid_train", losses, step)

        if (step + 1) % 100 == 0:
            elapsed = time.time() - start_time
            hybrid_trange.set_postfix({
                "Value Loss": f"{losses['value']:.4f}",
                "Q Loss": f"{losses['q']:.4f}",
                "Policy Loss": f"{losses['policy']:.4f}",
                "Elapsed": f"{elapsed:.2f}s"
            })

        if (step + 1) % 5000 == 0:
            print("Evaluating on benchmarks ...")
            eval_start = time.time()
            env_time = evaluate_benchmarks(agent, eval_env, step)
            print(f"Evaluation done in {time.time() - eval_start:.2f}s (env time: {env_time:.2f}s)")
            fl.flush()

        if (step + 1) % 2000 == 0:
            os.makedirs(cfg.results_dir, exist_ok=True)
            save_path = os.path.join(cfg.results_dir, f"iql_hybrid_step_{step}.pt")
            torch.save(agent.state_dict(), save_path)
            print(f"Checkpoint saved: {save_path}")

        state = next_state

    print(f"Hybrid fine-tuning finished in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    hybrid_finetune()
