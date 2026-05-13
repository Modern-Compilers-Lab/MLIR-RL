import warnings
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")

import argparse
import json
import logging
import re
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure as sb3_configure, HumanOutputFormat

from llm_action.src.env import MLIROptEnv
from llm_action.src.env.action_registry import load_action_registry

from llm_action.src.config import RL_RESULTS_DIR, MAX_STEPS, SB3_STDOUT_KEY_MAX_LENGTH

class FullEvalCallback(BaseCallback):
    """Run the policy on every benchmark with both greedy and sampling evaluation.

    Greedy: 1 deterministic run per benchmark (eval_greedy/ section).
    Sampling: K stochastic runs per benchmark (eval_sample/ section).
    Best model is saved based on greedy mean speedup.
    """

    def __init__(self, eval_env_fn, eval_freq: int = 1_000,
                 sample_runs: int = 5,
                 log_dir: str | Path = None, best_model_save_path: str | Path = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.eval_env_fn = eval_env_fn
        self.eval_freq = eval_freq
        self.sample_runs = sample_runs
        self.log_dir = Path(log_dir) if log_dir else None
        self.best_model_save_path = Path(best_model_save_path) if best_model_save_path else None
        self._eval_count = 0
        self.best_mean_speedup = -np.inf

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        self._eval_count += 1
        env = self.eval_env_fn()
        benchmarks = env.unwrapped.benchmarks
        n_benchmarks = len(benchmarks)

        # ── Greedy evaluation (1 deterministic run per benchmark) ──
        greedy_results = []
        for idx in range(n_benchmarks):
            greedy_results.append(self._run_episode(env, idx, deterministic=True))

        # ── Sampling evaluation (K stochastic runs per benchmark) ──
        sample_results = []  # list of lists: [benchmark_idx][run_k]
        for idx in range(n_benchmarks):
            runs = []
            for _ in range(self.sample_runs):
                runs.append(self._run_episode(env, idx, deterministic=False))
            sample_results.append(runs)

        self._log_and_save(greedy_results, sample_results)
        env.close()
        return True

    def _run_episode(self, env, benchmark_idx: int, deterministic: bool) -> dict:
        obs, info = env.reset(options={"benchmark_idx": benchmark_idx})
        done, truncated = False, False
        total_reward = 0.0

        while not (done or truncated):
            action_masks = env.action_masks()
            action, _ = self.model.predict(
                obs, deterministic=deterministic,
                action_masks=action_masks,
            )
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

        torch_t = info.get("torch_time_ms", -1)
        opt_t = info["opt_time_ms"]
        return {
            "benchmark": info["benchmark"],
            "reward": total_reward,
            "base_time_ms": info["base_time_ms"],
            "torch_time_ms": torch_t,
            "opt_time_ms": opt_t,
            "speedup": info["base_time_ms"] / opt_t if opt_t > 0 else 0.0,
            "speedup_to_torch": torch_t / opt_t if torch_t > 0 and opt_t > 0 else 0.0,
            "actions": info["action_history"],
            "n_steps": info["step"],
        }

    def _log_and_save(self, greedy_results: list[dict], sample_results: list[list[dict]]):
        timestep = self.num_timesteps
        metrics = {}

        def _geomean(values: list[float]) -> float:
            """Geometric mean of speedups; values <= 0 are clamped to 1.0 (no speedup)."""
            positive = [max(v, 1e-6) for v in values]
            return float(np.exp(np.mean(np.log(positive))))

        def _speedup_stats(speedups: list[float], prefix: str):
            """Log all/successful speedup aggregates under the given prefix."""
            successful = [s for s in speedups if s > 0]
            # All executions (failed = 0.0)
            metrics[f"{prefix}/mean_speedup"] = float(np.mean(speedups))
            metrics[f"{prefix}/geomean_speedup"] = _geomean(speedups)
            # Successful executions only
            if successful:
                metrics[f"{prefix}/mean_speedup_success"] = float(np.mean(successful))
                metrics[f"{prefix}/geomean_speedup_success"] = _geomean(successful)

        # ── Greedy metrics (eval_greedy/) ──
        greedy_speedups = [r["speedup"] for r in greedy_results]
        greedy_torch_speedups = [r["speedup_to_torch"] for r in greedy_results]
        greedy_rewards = [r["reward"] for r in greedy_results]

        metrics["eval_greedy/mean_reward"] = float(np.mean(greedy_rewards))
        _speedup_stats(greedy_speedups, "eval_greedy")
        _speedup_stats(greedy_torch_speedups, "eval_g_torch")

        for r in greedy_results:
            key = r["benchmark"].replace("/", "_")
            metrics[f"eval_greedy/speedup/{key}"] = r["speedup"]
            metrics[f"eval_greedy/torch_sp/{key}"] = r["speedup_to_torch"]
            metrics[f"eval_greedy/reward/{key}"] = r["reward"]

        # ── Sampling metrics (eval_sample/) ──
        # Flatten all runs across benchmarks for aggregate stats
        all_sample_speedups = []
        all_sample_torch_speedups = []
        all_sample_rewards = []
        for runs in sample_results:
            for r in runs:
                all_sample_speedups.append(r["speedup"])
                all_sample_torch_speedups.append(r["speedup_to_torch"])
                all_sample_rewards.append(r["reward"])

        metrics["eval_sample/mean_reward"] = float(np.mean(all_sample_rewards))
        metrics["eval_sample/std_reward"] = float(np.std(all_sample_rewards))
        metrics["eval_sample/min_reward"] = float(np.min(all_sample_rewards))
        metrics["eval_sample/max_reward"] = float(np.max(all_sample_rewards))
        _speedup_stats(all_sample_speedups, "eval_sample")
        _speedup_stats(all_sample_torch_speedups, "eval_s_torch")
        metrics["eval_sample/std_speedup"] = float(np.std(all_sample_speedups))
        metrics["eval_sample/min_speedup"] = float(np.min(all_sample_speedups))
        metrics["eval_sample/max_speedup"] = float(np.max(all_sample_speedups))

        # Per-benchmark sampling stats (mean over K runs)
        for runs in sample_results:
            key = runs[0]["benchmark"].replace("/", "_")
            sp = [r["speedup"] for r in runs]
            sp_torch = [r["speedup_to_torch"] for r in runs]
            rw = [r["reward"] for r in runs]
            metrics[f"eval_sample/speedup/{key}"] = float(np.mean(sp))
            metrics[f"eval_sample/torch_sp/{key}"] = float(np.mean(sp_torch))
            metrics[f"eval_sample/reward/{key}"] = float(np.mean(rw))
            metrics[f"eval_sample/best_sp/{key}"] = float(np.max(sp))

        wandb.log(metrics)
        for k, v in metrics.items():
            self.logger.record(k, v)

        # Best model saving (based on greedy speedup — the deterministic objective)
        greedy_mean_speedup = float(np.mean(greedy_speedups))
        if greedy_mean_speedup > self.best_mean_speedup:
            self.best_mean_speedup = greedy_mean_speedup
            if self.best_model_save_path:
                self.best_model_save_path.mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.best_model_save_path / "best_model"))
                logging.info(f"[Eval #{self._eval_count}] New best greedy speedup: {greedy_mean_speedup:.4f}x — model saved")

        # JSON log
        if self.log_dir:
            eval_dir = self.log_dir / "per_benchmark_evals"
            eval_dir.mkdir(parents=True, exist_ok=True)
            out_path = eval_dir / f"eval_{self._eval_count:04d}_ts{timestep}.json"
            payload = {
                "timestep": timestep,
                "eval_count": self._eval_count,
                "best_mean_speedup": self.best_mean_speedup,
                "greedy_results": greedy_results,
                "sample_results": sample_results,
                "summary": {k: float(v) for k, v in metrics.items()
                            if "/speedup/" not in k and "/reward/" not in k and "/best_speedup/" not in k},
            }
            out_path.write_text(json.dumps(payload, indent=2))

        if self.verbose:
            sample_mean_sp = float(np.mean(all_sample_speedups))
            logging.info(
                f"[Eval #{self._eval_count}] ts={timestep} | "
                f"greedy_speedup={greedy_mean_speedup:.3f}x | "
                f"sample_speedup={sample_mean_sp:.3f}x (K={self.sample_runs}) | "
                f"benchmarks={len(greedy_results)}"
            )

class MLIRMetricsCallback(BaseCallback):
    def __init__(self, action_names: list[str] | None = None, total_timesteps: int = 0):
        super().__init__()
        self._action_names = action_names or []
        self._total_timesteps = total_timesteps

        # Existing rolling windows
        self._ep_rewards: list[float] = []
        self._ep_speedups: list[float] = []
        self._action_counts: defaultdict[str, int] = defaultdict(int)

        # Success rate tracking
        self._ep_success_rates: list[float] = []
        self._per_action_success: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])
        self._per_action_success_rollout: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])

        # Per-benchmark tracking (rollout window)
        self._per_bench_speedups: defaultdict[str, list[float]] = defaultdict(list)
        self._per_bench_rewards: defaultdict[str, list[float]] = defaultdict(list)

        # Convergence indicators
        self._reward_window: deque[float] = deque(maxlen=100)
        self._speedup_window: deque[float] = deque(maxlen=100)
        self._prev_mean_speedup: float = 0.0
        self._speedup_plateau_count: int = 0
        self._prev_entropy: float | None = None

        # ETA / throughput
        self._start_time: float = 0.0
        self._ep_count: int = 0

    def _on_training_start(self):
        self._start_time = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue

            ep = info["episode"]
            self._ep_rewards.append(ep["r"])
            self._reward_window.append(ep["r"])
            self._ep_count += 1

            history = info.get("action_history", [])
            outcomes = info.get("action_outcomes", [])
            terminal_reward = info.get("reward", 0.0)

            base_t = info.get("base_time_ms", -1)
            opt_t = info.get("opt_time_ms", -1)

            # Per-episode success rate
            n_total = len(outcomes)
            n_success = sum(1 for _, ok in outcomes if ok)
            ep_success_rate = n_success / n_total if n_total > 0 else 0.0
            self._ep_success_rates.append(ep_success_rate)

            # Per-action success tallies
            for action_name, ok in outcomes:
                self._per_action_success[action_name][1] += 1
                self._per_action_success_rollout[action_name][1] += 1
                if ok:
                    self._per_action_success[action_name][0] += 1
                    self._per_action_success_rollout[action_name][0] += 1

            # episode_reward = cumulative reward (sum of all steps; in final mode same as last step)
            # last_step_reward = reward from the final step only (terminal or intermediate)
            metrics = {
                "rl/episode_reward": ep["r"],
                "rl/episode_length": ep["l"],
                "rl/last_step_reward": terminal_reward,
                "rl/n_successful_actions": n_success,
                "rl/n_failed_actions": n_total - n_success,
                "rl_success/ep_success_rate": ep_success_rate,
                "rl_success/ep_n_total_actions": n_total,
            }

            if base_t > 0:
                metrics["rl/base_time_ms"] = base_t
            if opt_t > 0:
                metrics["rl/opt_time_ms"] = opt_t

            # Log speedup for every episode (0.0 when execution fails) to avoid survivorship bias
            if base_t > 0 and opt_t > 0:
                speedup = base_t / opt_t
            else:
                speedup = 0.0
            self._ep_speedups.append(speedup)
            self._speedup_window.append(speedup)
            metrics["rl/speedup"] = speedup

            # Per-benchmark accumulation
            bench_name = info.get("benchmark", "")
            if bench_name:
                self._per_bench_speedups[bench_name].append(speedup)
                self._per_bench_rewards[bench_name].append(ep["r"])

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

        # ── Existing rolling averages ──
        if self._ep_rewards:
            metrics["rl/mean_reward_100ep"] = np.mean(self._ep_rewards[-100:])
            self._ep_rewards = self._ep_rewards[-100:]
        if self._ep_speedups:
            metrics["rl/mean_speedup_100ep"] = np.mean(self._ep_speedups[-100:])
            self._ep_speedups = self._ep_speedups[-100:]

        total = sum(self._action_counts.values()) or 1
        for name in self._action_names:
            metrics[f"rl_actions/{name}"] = self._action_counts.get(name, 0) / total

        # ── Per-benchmark breakdown ──
        for bench, sp_list in self._per_bench_speedups.items():
            if sp_list:
                key = bench.replace("/", "_")
                metrics[f"rl_bench/sp/{key}"] = float(np.mean(sp_list))
                metrics[f"rl_bench/rw/{key}"] = float(np.mean(self._per_bench_rewards.get(bench, [0])))
        self._per_bench_speedups = defaultdict(list)
        self._per_bench_rewards = defaultdict(list)

        # ── Success rate metrics ──
        if self._ep_success_rates:
            metrics["rl_success/mean_success_rate_100ep"] = np.mean(self._ep_success_rates[-100:])
            self._ep_success_rates = self._ep_success_rates[-100:]

        for name in self._action_names:
            counts = self._per_action_success_rollout.get(name)
            if counts and counts[1] > 0:
                metrics[f"rl_success/{name}"] = counts[0] / counts[1]
        self._per_action_success_rollout = defaultdict(lambda: [0, 0])

        # ── Convergence indicators ──
        if len(self._reward_window) > 1:
            mean_r = np.mean(self._reward_window)
            std_r = np.std(self._reward_window)
            n = len(self._reward_window)
            ci = 1.96 * std_r / np.sqrt(n)
            metrics["rl_convergence/reward_std_100ep"] = std_r
            metrics["rl_convergence/reward_ci_lower_100ep"] = mean_r - ci
            metrics["rl_convergence/reward_ci_upper_100ep"] = mean_r + ci

        if self._speedup_window:
            mean_sp = np.mean(self._speedup_window)
            delta = mean_sp - self._prev_mean_speedup
            metrics["rl_convergence/speedup_delta"] = delta
            if self._prev_mean_speedup > 0 and abs(delta) / self._prev_mean_speedup < 0.005:
                self._speedup_plateau_count += 1
            else:
                self._speedup_plateau_count = 0
            metrics["rl_convergence/speedup_plateau_count"] = self._speedup_plateau_count
            self._prev_mean_speedup = mean_sp

        # Entropy decay
        current_entropy = None
        if hasattr(self.logger, "name_to_value"):
            current_entropy = self.logger.name_to_value.get("train/entropy_loss")
        if current_entropy is not None:
            if self._prev_entropy is not None and abs(self._prev_entropy) > 1e-8:
                metrics["rl_convergence/entropy_decay_rate"] = (self._prev_entropy - current_entropy) / abs(self._prev_entropy)
            self._prev_entropy = current_entropy

        # ── ETA / throughput ──
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            steps_per_sec = self.num_timesteps / elapsed
            metrics["rl_progress/steps_per_sec"] = steps_per_sec
            metrics["rl_progress/episodes_per_sec"] = self._ep_count / elapsed
            metrics["rl_progress/wall_clock_elapsed_min"] = elapsed / 60.0

            if self._total_timesteps > 0:
                metrics["rl_progress/pct_complete"] = 100.0 * self.num_timesteps / self._total_timesteps
                remaining = self._total_timesteps - self.num_timesteps
                if steps_per_sec > 0:
                    metrics["rl_progress/eta_minutes"] = remaining / steps_per_sec / 60.0

        for k, v in metrics.items():
            self.logger.record(k, v)

        # Forward SB3 training metrics (train/*, rollout/*) to wandb
        train_metrics = {}
        if hasattr(self.logger, "name_to_value"):
            for k, v in self.logger.name_to_value.items():
                if isinstance(v, (int, float)):
                    train_metrics[k] = v

        wandb.log({**metrics, **train_metrics})

class EntCoefScheduleCallback(BaseCallback):
    """Anneal `model.ent_coef` from `initial` to `final` over `total_episodes`.

    Progress is measured in completed episodes (counted via the SB3 Monitor
    `info["episode"]` signal, matching the pattern used by MLIRMetricsCallback),
    so the schedule is consistent across runs with different per-episode step
    counts. The value is committed to `model.ent_coef` at every rollout-start;
    PPO reads the attribute fresh inside each minibatch update, so the new
    value applies to the next gradient update without subclassing the algo.
    Logs `train/ent_coef` (forwarded to W&B by the existing pipeline).
    """
    def __init__(self, initial: float, final: float, total_episodes: int,
                 schedule: str = "linear", verbose: int = 0):
        super().__init__(verbose)
        assert schedule in ("linear", "exponential")
        if schedule == "exponential" and (initial <= 0 or final <= 0):
            raise ValueError("exponential schedule requires initial > 0 and final > 0")
        self.initial = float(initial)
        self.final = float(final)
        self.total_episodes = max(1, int(total_episodes))
        self.schedule = schedule
        self._ep_count: int = 0

    def _compute(self) -> float:
        p = min(self._ep_count / self.total_episodes, 1.0)
        if self.schedule == "linear":
            return self.initial + (self.final - self.initial) * p
        return self.initial * (self.final / self.initial) ** p

    def _on_rollout_start(self) -> None:
        new = self._compute()
        self.model.ent_coef = new
        self.logger.record("train/ent_coef", new)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._ep_count += 1
        return True

def parse_args():
    p = argparse.ArgumentParser(description="PPO for MLIR optimization")
    p.add_argument("--benchmarks-name", type=str, default="standard", help="Name of the benchmark set under data/benchmarks/ (default: standard)")
    p.add_argument("--total-timesteps", type=int, default=250_000) # 500_000
    p.add_argument("--n-steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--n-epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=1e-2)
    p.add_argument("--ent-coef-final", type=float, default=1e-4)
    p.add_argument("--ent-coef-schedule", type=str, default="linear", choices=["linear", "exponential"])
    p.add_argument("--vf-coef", type=float, default=0.005)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--net-arch", type=int, nargs="+", default=[512, 512, 512])
    p.add_argument("--n-envs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=MAX_STEPS)
    p.add_argument("--action-version", type=str, default="v10")
    p.add_argument("--param-mode", type=str, default="multidiscrete", choices=["multidiscrete", "two_policy", "llm"])
    p.add_argument("--executor-type", type=str, default="dask", choices=["slurm", "dask", "local"])
    p.add_argument("--dask-nodes", type=int, default=2)
    p.add_argument("--history-mode", type=str, default="success-encoding", choices=["include-all", "ignore-failed", "success-encoding"])
    p.add_argument("--reward-scale", type=str, default="log", choices=["log", "raw", "delta"])
    p.add_argument("--reward-mode", type=str, default="final", choices=["final", "intermediate", "schedule"])
    p.add_argument("--reward-baseline", type=str, default="mlir", choices=["mlir", "torch"], help="Baseline for speedup ratio: 'mlir' (unoptimized MLIR) or 'torch' (PyTorch)")
    p.add_argument("--checkpoint-freq", type=int, default=5_000)
    p.add_argument("--eval-freq", type=int, default=1_000)
    p.add_argument("--eval-sample-runs", type=int, default=5, help="Number of stochastic sampling runs per benchmark during evaluation")
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--exp-name", "-n", type=str, default=None, help="Free-form experiment label appended to the run name")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", type=str, default="mlir-rl")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--resume", type=str, default=None, help="Path to a checkpoint .zip to resume training from")
    p.add_argument("--enable-dependency-masking", action=argparse.BooleanOptionalAction, default=True, help="Mask actions made provably illegal by registry.ACTION_DEPENDENCIES (default: enabled). Pass --no-enable-dependency-masking to disable.")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "httpcore", "anthropic", "agno", "openai", "groq",
                   "distributed", "distributed.scheduler", "distributed.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # When resuming, reuse the log directory from the checkpoint path
    if args.resume:
        checkpoint_path = Path(args.resume)
        log_dir = checkpoint_path.parent.parent  # checkpoints/ -> run_dir/
        run_name = log_dir.name
        logging.info(f"Resuming from checkpoint: {args.resume}")
    else:
        run_name = f"ppo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.exp_name:
            label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.exp_name.strip()).strip("_")
            if label:
                run_name = f"{run_name}_{label}"
        log_dir = Path(args.log_dir) if args.log_dir else RL_RESULTS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
    logging.info(f"Run: {log_dir}")

    wandb_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        dir=str(log_dir),
    )
    if args.resume:
        # Try to resume the previous wandb run for this log directory
        wandb_id_file = log_dir / "wandb_run_id.txt"
        if wandb_id_file.exists():
            wandb_kwargs["id"] = wandb_id_file.read_text().strip()
            wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)
    # Persist wandb run ID for future resumes
    (log_dir / "wandb_run_id.txt").write_text(wandb.run.id)

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
        "history_mode": args.history_mode,
        "reward_scale": args.reward_scale,
        "reward_mode": args.reward_mode,
        "reward_baseline": args.reward_baseline,
        "enable_dependency_masking": args.enable_dependency_masking,
        "verbose": args.verbose,
    }

    def _make_env(config):
        def _init():
            env = MLIROptEnv(config=config)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    train_env = make_vec_env(
        _make_env({**env_config, "benchmarks_split": "train"}),
        n_envs=args.n_envs, seed=args.seed,
    )

    sb3_logger = sb3_configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    for fmt in sb3_logger.output_formats:
        if isinstance(fmt, HumanOutputFormat):
            fmt.max_length = SB3_STDOUT_KEY_MAX_LENGTH
            break

    if args.resume:
        model = MaskablePPO.load(args.resume, env=train_env)
        logging.info(f"Loaded model from {args.resume}")
    else:
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

    def _make_eval_env():
        env = MLIROptEnv(config={**env_config, "benchmarks_split": "eval"})
        return ActionMasker(env, lambda e: e.action_masks())

    callbacks = [
        MLIRMetricsCallback(action_names=action_names, total_timesteps=args.total_timesteps),
        CheckpointCallback(save_freq=args.checkpoint_freq,
                           save_path=str(log_dir / "checkpoints"), name_prefix="ppo_mlir"),
        FullEvalCallback(
            eval_env_fn=_make_eval_env,
            eval_freq=args.eval_freq,
            sample_runs=args.eval_sample_runs,
            log_dir=log_dir,
            best_model_save_path=log_dir / "best_model",
            verbose=1,
        ),
    ]
    if args.ent_coef_final is not None:
        total_episodes = max(1, args.total_timesteps // max(1, args.max_steps))
        callbacks.append(EntCoefScheduleCallback(
            initial=args.ent_coef,
            final=args.ent_coef_final,
            total_episodes=total_episodes,
            schedule=args.ent_coef_schedule,
        ))

    logging.info("Starting PPO training...")
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=True,
                use_masking=True)
    model.save(str(log_dir / "final_model"))
    logging.info(f"Done. Model saved to {log_dir / 'final_model'}")

    wandb.finish()
    train_env.close()

    if args.executor_type == "dask":
        from llm_action.src.execution.dask_executor import close_shared_client
        close_shared_client()

if __name__ == "__main__":
    main()
