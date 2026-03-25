import warnings
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as sb3_configure

from llm_action.src.env import MLIROptEnv
from llm_action.src.env.action_registry import load_action_registry

from llm_action.src.config import RL_RESULTS_DIR, MAX_STEPS

class MLIREvalCallback(MaskableEvalCallback):
    """EvalCallback that also logs eval metrics to wandb under eval/ prefix."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_speedups: list[float] = []

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        super()._log_success_callback(locals_, globals_)
        info = locals_.get("info", {})
        if isinstance(info, dict) and locals_.get("done", False):
            base_t = info.get("base_time_ms", -1)
            opt_t = info.get("opt_time_ms", -1)
            if base_t > 0 and opt_t > 0:
                self._eval_speedups.append(base_t / opt_t)

    def _on_step(self) -> bool:
        self._eval_speedups.clear()
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0 and self.last_mean_reward is not None:
            metrics = {
                "eval/mean_reward": self.last_mean_reward,
            }

            # Extract per-episode eval info from the evaluations log
            if hasattr(self, 'evaluations_results') and len(self.evaluations_results) > 0:
                last_eval = self.evaluations_results[-1]
                metrics["eval/min_reward"] = float(np.min(last_eval))
                metrics["eval/max_reward"] = float(np.max(last_eval))
                metrics["eval/std_reward"] = float(np.std(last_eval))

            if self._eval_speedups:
                metrics["eval/mean_speedup"] = float(np.mean(self._eval_speedups))
                metrics["eval/min_speedup"] = float(np.min(self._eval_speedups))
                metrics["eval/max_speedup"] = float(np.max(self._eval_speedups))

            for k, v in metrics.items():
                self.logger.record(k, v)
            wandb.log(metrics)

        return result

class MLIRMetricsCallback(BaseCallback):
    def __init__(self, action_names: list[str] | None = None):
        super().__init__()
        self._action_names = action_names or []
        self._ep_rewards = []
        self._ep_speedups = []
        self._action_counts = defaultdict(int)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue

            ep = info["episode"]
            self._ep_rewards.append(ep["r"])

            history = info.get("action_history", [])
            terminal_reward = info.get("reward", 0.0)

            base_t = info.get("base_time_ms", -1)
            opt_t = info.get("opt_time_ms", -1)

            metrics = {
                "rl/episode_reward": ep["r"],
                "rl/episode_length": ep["l"],
                "rl/n_successful_actions": len(history),
                "rl/n_failed_actions": max(0, ep["l"] - len(history) - 1),
                "rl/terminal_reward": terminal_reward,
            }

            if base_t > 0:
                metrics["rl/base_time_ms"] = base_t
            if opt_t > 0:
                metrics["rl/opt_time_ms"] = opt_t

            if base_t > 0 and opt_t > 0:
                speedup = base_t / opt_t
                self._ep_speedups.append(speedup)
                metrics["rl/speedup"] = speedup

            for name in history:
                self._action_counts[name] += 1

            # SB3 logger (flushed on rollout_end -> TB/CSV)
            for k, v in metrics.items():
                self.logger.record(k, v)

            # wandb: log immediately so charts update in real time
            wandb.log(metrics)

        return True

    def _on_rollout_end(self):
        metrics = {}

        if self._ep_rewards:
            metrics["rl/mean_reward_100ep"] = np.mean(self._ep_rewards[-100:])
            self._ep_rewards = self._ep_rewards[-100:]
        if self._ep_speedups:
            metrics["rl/mean_speedup_100ep"] = np.mean(self._ep_speedups[-100:])
            self._ep_speedups = self._ep_speedups[-100:]

        total = sum(self._action_counts.values()) or 1
        for name in self._action_names:
            metrics[f"rl_actions/{name}"] = self._action_counts.get(name, 0) / total

        for k, v in metrics.items():
            self.logger.record(k, v)

        # Forward SB3 training metrics (train/*, rollout/*) to wandb
        # SB3 records them before calling _on_rollout_end
        train_metrics = {}
        if hasattr(self.logger, "name_to_value"):
            for k, v in self.logger.name_to_value.items():
                if isinstance(v, (int, float)):
                    train_metrics[k] = v

        wandb.log({**metrics, **train_metrics})

def parse_args():
    p = argparse.ArgumentParser(description="PPO for MLIR optimization")
    p.add_argument("--benchmarks-name", type=str, default="matmul", help="Name of the benchmarks subdirectory under data/benchmarks/")
    p.add_argument("--total-timesteps", type=int, default=500_000)
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--vf-coef", type=float, default=0.005)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net-arch", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--action-version", type=str, default="v9")
    p.add_argument("--param-mode", type=str, default="multidiscrete",
                    choices=["multidiscrete", "two_policy", "llm"])
    p.add_argument("--executor-type", type=str, default="dask",
                    choices=["slurm", "dask", "local"])
    p.add_argument("--dask-nodes", type=int, default=4)
    p.add_argument("--reward-scale", type=str, default="raw", choices=["log", "raw", "delta"])
    p.add_argument("--reward-mode", type=str, default="final", choices=["final", "intermediate", "schedule"])
    p.add_argument("--checkpoint-freq", type=int, default=5_000)
    p.add_argument("--eval-freq", type=int, default=1_000)
    p.add_argument("--n-eval-episodes", type=int, default=5)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", type=str, default="mlir-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "httpcore", "anthropic", "agno", "openai", "groq",
                   "distributed", "distributed.scheduler", "distributed.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = Path(args.log_dir) if args.log_dir else RL_RESULTS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    logging.info(f"Run: {log_dir}")

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        dir=str(log_dir),
    )

    if args.executor_type == "dask":
        from llm_action.src.execution.dask_executor import init_shared_client
        init_shared_client(args.dask_nodes)
        logging.info(f"Dask cluster ready with {args.dask_nodes} workers")

    env_config = {
        "benchmarks_name": args.benchmarks_name,
        "action_version": args.action_version,
        "param_mode": args.param_mode,
        "executor_type": args.executor_type,
        "max_steps": args.max_steps,
        "reward_scale": args.reward_scale,
        "reward_mode": args.reward_mode,
        "verbose": args.verbose,
    }

    def _make_env(config):
        def _init():
            env = MLIROptEnv(config=config)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    train_env = make_vec_env(_make_env(env_config), n_envs=args.n_envs, seed=args.seed)
    eval_env = make_vec_env(_make_env(env_config), n_envs=1, seed=args.seed + 1000)

    sb3_logger = sb3_configure(str(log_dir), ["stdout", "csv", "tensorboard"])

    model = MaskablePPO(
        "MlpPolicy", train_env,
        learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
        n_epochs=args.n_epochs, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_range=args.clip_range, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs={"net_arch": args.net_arch},
        seed=args.seed, verbose=1,
    )
    model.set_logger(sb3_logger)

    registry = load_action_registry(args.action_version)
    action_names = [cls.__name__ for cls in registry.action_classes]

    callbacks = [
        MLIRMetricsCallback(action_names=action_names),
        CheckpointCallback(save_freq=args.checkpoint_freq,
                           save_path=str(log_dir / "checkpoints"), name_prefix="ppo_mlir"),
        MLIREvalCallback(eval_env, best_model_save_path=str(log_dir / "best_model"),
                         log_path=str(log_dir / "eval_logs"),
                         eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
                         deterministic=True, use_masking=True),
    ]

    logging.info("Starting PPO training...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True,
                use_masking=True)
    model.save(str(log_dir / "final_model"))
    logging.info(f"Done. Model saved to {log_dir / 'final_model'}")

    wandb.finish()
    train_env.close()
    eval_env.close()

    if args.executor_type == "dask":
        from llm_action.src.execution.dask_executor import close_shared_client
        close_shared_client()

if __name__ == "__main__":
    main()
