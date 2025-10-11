from dotenv import load_dotenv
load_dotenv(override=True)

import os
import torch
import numpy as np

from rl_autoschedular.env import Env
from rl_autoschedular.observation import Observation
from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.model import HiearchyModel


def run_inference(model_ckpt: str, bench_idx: int = 10, repeat: int = 5):
    # === Create evaluation environment ===
    env = Env(is_training=False)
    state = env.reset(bench_idx=bench_idx)

    # === Load model ===
    model = HiearchyModel()
    checkpoint = torch.load(model_ckpt, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)  # allow partial load
    model.eval()

    print(f"Loaded checkpoint: {model_ckpt}")
    print(f"Running inference on benchmark index {bench_idx}")

    seq = []
    bench_done = False
    total_reward = 0.0
    acc_values = []

    while not bench_done:
        obs = Observation.from_state(state)

        with torch.no_grad():
            dists = model.policy_model(obs)
            action_index = ActionSpace.sample(obs, dists, dists, greedy=True)

        action = ActionSpace.action_by_index(action_index[0], state)
        seq.append(action)

        # repeat execution for stability
        tmp_rewards, tmp_accs = [], []
        for _ in range(repeat):
            next_state, reward, op_done, acc = env.step(state, action)
            tmp_rewards.append(reward)
            if acc is not None:
                tmp_accs.append(acc)

        # use median to avoid outliers
        reward = np.median(tmp_rewards)
        acc = np.median(tmp_accs) if tmp_accs else None

        total_reward += reward

        if op_done:
            next_state, bench_done = env.get_next_op_state(next_state)

        state = next_state

        if acc is not None:
            acc_values.append(acc)

    avg_acc = np.median(acc_values) if acc_values else None

    print("\n=== Inference finished ===")
    print("Sequence of actions:", seq)
    print("Total reward:", total_reward)
    print("Median acceleration:", avg_acc)


if __name__ == "__main__":
    model_ckpt = "./tests/checkpoints/model.pth"  # adjust path
    run_inference(model_ckpt, bench_idx=0, repeat=5)
