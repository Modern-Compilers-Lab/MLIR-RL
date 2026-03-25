import argparse

import numpy as np
from llm_action.src.env import MLIROptEnv

from llm_action.src.config import L, MAX_STEPS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--action-version", default="v10")
    p.add_argument("--benchmarks-name", default="matmul")
    p.add_argument("--executor-type", default="local", choices=["local", "dask", "slurm"])
    p.add_argument("--param-mode", default="multidiscrete", choices=["multidiscrete", "two_policy", "llm"])
    args = p.parse_args()

    config = {
        "action_version": args.action_version,
        "param_mode": args.param_mode,
        "executor_type": args.executor_type,
        "max_steps": MAX_STEPS,
        "benchmarks_name": args.benchmarks_name,
        "verbose": False,
    }

    print(f"=== MLIROptEnv Lifecycle ({args.action_version}, {args.param_mode}, {args.executor_type}) ===")
    print(f"  Config: {config}")
    print()

    print("--- Environment creation ---")
    env = MLIROptEnv(config=config)
    print(f"  Actions: {env.registry.num_actions} + done = {env.registry.total_actions}")
    print(f"  Benchmarks: {len(env.benchmarks)} [{', '.join(b.name for b in env.benchmarks[:5])}{'...' if len(env.benchmarks) > 5 else ''}]")
    print(f"  Action space: {env.action_space}")
    print(f"  Obs space: {env.observation_space.shape}")
    if args.param_mode == "multidiscrete":
        print(f"  MultiDiscrete nvec: {list(env.action_space.nvec)}")
    print()

    print("--- Reset ---")
    obs, info = env.reset(seed=42)
    print(f"  Benchmark: {info['benchmark']}")
    print(f"  Baseline: {env._benchmark.base_exec_time_ms:.2f} ms")
    print(f"  n_loops: {env._n_loops}")
    print(f"  Obs: shape={obs.shape}, dtype={obs.dtype}, nonzero={(obs != 0).sum()}")
    assert env.observation_space.contains(obs), "FAIL: obs out of bounds"
    assert env._n_loops >= 1, "FAIL: n_loops < 1"
    print("  # Valid observation\n")

    print("--- Action masks ---")
    masks = env.action_masks()
    print(f"  Mask: length={len(masks)}, dtype={masks.dtype}, True={masks.sum()}/{len(masks)}")
    assert masks[env.registry.done_idx], "FAIL: done should be unmasked"
    print("  # Done always available\n")

    print("--- Step (random action) ---")
    action = env.action_space.sample()
    if args.param_mode == "multidiscrete":
        print(f"  Sampled action: {list(np.asarray(action, dtype=int))}")
    else:
        print(f"  Sampled action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  reward={reward:.4f}, terminated={terminated}, truncated={truncated}, failed={info.get('failed', False)}")
    print(f"  Obs: shape={obs.shape}, nonzero={(obs != 0).sum()}")
    assert env.observation_space.contains(obs), "FAIL: obs out of bounds after step"
    print("  # Valid step\n")

    if not terminated:
        print("--- Done action ---")
        if args.param_mode == "multidiscrete":
            done = np.zeros(len(env.action_space.nvec), dtype=int)
            done[0] = env.registry.done_idx
        else:
            done = env.registry.done_idx
        obs, reward, terminated, truncated, info = env.step(done)
        print(f"  reward={reward:.4f}, terminated={terminated}")
        assert terminated, "FAIL: done should terminate"
        print("  # Episode terminated\n")

    print("--- Truncation (max_steps) ---")
    env.reset(seed=123)
    steps = 0
    for i in range(config["max_steps"]):
        if args.param_mode == "multidiscrete":
            action = np.zeros(len(env.action_space.nvec), dtype=int)
            action[0] = 0
        else:
            action = 0
        _, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated:
            print(f"  Terminated early at step {steps} (action may have consumed tag)")
            break
    else:
        assert truncated, "FAIL: should truncate at max_steps"
        print(f"  Truncated at step {steps} #")
    print()

    print("--- Unique actions masking ---")
    env.reset(seed=42)
    if args.param_mode == "multidiscrete":
        action = np.zeros(len(env.action_space.nvec), dtype=int)
        action[0] = 0
    else:
        action = 0
    env.step(action)
    masks = env.action_masks()
    print(f"  After using action 0: mask[0]={masks[0]}, mask[done]={masks[env.registry.done_idx]}")
    assert masks[env.registry.done_idx], "FAIL: done should always be available"
    print("  # Done still available after action use\n")

    env.close()
    print("=== ALL CHECKS PASSED ===")

if __name__ == "__main__":
    main()
