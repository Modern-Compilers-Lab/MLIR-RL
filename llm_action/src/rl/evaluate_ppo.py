import warnings
warnings.filterwarnings("ignore", message="urllib3.*doesn't match a supported version")

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from llm_action.src.env import MLIROptEnv
# Ensure BehaviorMaskedActorCriticPolicy is importable so MaskablePPO.load can restore
# models trained with --policy-mask-mode behavior-only (policy class resolved by ref).
import llm_action.src.rl.behavior_masking  # noqa: F401
from llm_action.src.config import RL_RESULTS_DIR, EVALUATION_RESULTS_DIR

ENV_CONFIG_KEYS = [
    "benchmarks_name", "action_version", "param_mode", "max_steps",
    "history_mode", "reward_scale", "reward_mode", "reward_baseline",
    "loop_bound_encoding", "enable_dependency_masking", "masking_mode",
]

STAT_KEYS = ["min", "q25", "median", "q75", "max"]

# Kernels to skip during evaluation (excluded from CSVs, results.json, and live runs).
EXCLUDED_KERNELS = {"relu_256_10"}

PER_KERNEL_METRICS = ["speedup", "exec_time_ms", "speedup_to_torch"]

SUMMARY_AGGS = {
    "speedup": ["mean", "geomean"],
    "exec_time_ms": ["mean"],
    "speedup_to_torch": ["mean", "geomean"],
}

def _geomean(values: list[float]) -> float:
    """Geometric mean of speedups; values <= 0 are clamped to 1e-6 (failures)."""
    positive = [max(v, 1e-6) for v in values]
    return float(np.exp(np.mean(np.log(positive))))

def _category(benchmark: str) -> str:
    """Operation family = the leading non-numeric tokens of the kernel name.

    Joins `_`-separated tokens up to (excluding) the first all-digit token, e.g.
    `conv_2d_nchw_fchw_256_..` -> `conv_2d_nchw_fchw`, `matmul_256_1024_1024` ->
    `matmul`, `relu_256_10` -> `relu`. Matches the old `mlir_rl/per_kernel.csv`.
    """
    tokens = benchmark.split("_")
    prefix = []
    for tok in tokens:
        if tok.isdigit():
            break
        prefix.append(tok)
    return "_".join(prefix) if prefix else benchmark

def run_episode(model, env, benchmark_idx: int, deterministic: bool) -> dict:
    """Run one episode on a benchmark; mirrors FullEvalCallback._run_episode."""
    obs, info = env.reset(options={"benchmark_idx": benchmark_idx})
    done, truncated = False, False
    total_reward = 0.0

    while not (done or truncated):
        action_masks = env.action_masks()
        action, _ = model.predict(
            obs, deterministic=deterministic, action_masks=action_masks,
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

def _five_number(values: list[float]) -> dict:
    """5-number summary as ACTUAL order statistics (no interpolation).

    Positions are the sorted ranks round(q*(n-1)) for q in {0,.25,.5,.75,1}; for
    the standard n=9 this is the 1st/3rd/5th/7th/9th value, so every reported
    number is a real measurement.
    """
    s = sorted(values)
    n = len(s)
    return {k: float(s[int(round(q * (n - 1)))])
            for k, q in zip(STAT_KEYS, (0.0, 0.25, 0.5, 0.75, 1.0))}

def _kernel_stats(greedy_runs: list[dict]) -> dict:
    """Per-metric 5-number summary over the deterministic greedy runs.

    The greedy schedule is identical across runs; only the measured execution
    time varies, which propagates to speedup and speedup_to_torch.
    """
    return {
        "speedup": _five_number([r["speedup"] for r in greedy_runs]),
        "exec_time_ms": _five_number([r["opt_time_ms"] for r in greedy_runs]),
        "speedup_to_torch": _five_number([r["speedup_to_torch"] for r in greedy_runs]),
    }

def _best_eval(run_dir) -> dict:
    """The training eval whose model was saved as best_model.

    FullEvalCallback saves best_model at the eval with the highest arithmetic-mean
    greedy speedup, so we return the per_benchmark_evals JSON maximizing that mean
    (ties -> earliest). Raises if no eval logs exist.
    """
    files = sorted(Path(run_dir).glob("per_benchmark_evals/eval_*.json"))
    if not files:
        raise FileNotFoundError(
            f"No per_benchmark_evals/*.json in {run_dir}; training-logs mode needs them.")
    best, best_mean = None, -np.inf
    for f in files:
        d = json.loads(f.read_text())
        m = float(np.mean([r["speedup"] for r in d["greedy_results"]]))
        if m > best_mean:
            best_mean, best = m, d
    return best

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO agent on the eval benchmarks")
    p.add_argument("--run-name", type=str, required=True,
                   help="Full training run directory name under results/rl/ "
                        "(e.g. ppo_20260520_225443_v35_dataset_add). Unique via its datetime.")
    p.add_argument("--mode", type=str, default="execution", choices=["execution", "training-logs"],
                   help="execution: load best_model and re-run live (9-run 5-number stats). "
                        "training-logs: pull the best model's eval straight from the SB3 "
                        "per_benchmark_evals/ logs (single measurement per kernel, no model/env/dask). "
                        "executor/greedy-runs/sampling args are ignored in training-logs mode.")
    p.add_argument("--executor-type", type=str, default="dask", choices=["slurm", "dask", "local"])
    p.add_argument("--dask-nodes", type=int, default=1)
    p.add_argument("--eval-greedy-runs", type=int, default=9,
                   help="Number of deterministic greedy runs per benchmark; a 5-number "
                        "summary (min/q25/median/q75/max as actual order statistics) of each "
                        "metric is reported. Default 9 -> stats land on real measurements.")
    p.add_argument("--enable-sampling", action=argparse.BooleanOptionalAction, default=False,
                   help="Also run K stochastic rollouts per benchmark, recorded in results.json "
                        "only (CSVs stay greedy-stats only). Default: disabled.")
    p.add_argument("--eval-sample-runs", type=int, default=5,
                   help="Number of stochastic sampling runs per benchmark when --enable-sampling")
    p.add_argument("--out-name", type=str, default=None,
                   help="Output sub-directory under results/evaluation/ (default: the run name)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "httpcore", "anthropic", "agno", "openai", "groq",
                  "distributed", "distributed.scheduler", "distributed.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ── Locate the run and its best model ──
    run_dir = RL_RESULTS_DIR / args.run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in run dir: {config_path}")
    train_config = json.loads(config_path.read_text())
    out_name = args.out_name or args.run_name
    out_dir = EVALUATION_RESULTS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Evaluating run: {run_dir}  (mode={args.mode})")

    # ── training-logs mode: pull the best model's eval from the SB3 logs ──
    # (no model load, no env, no dask — single measurement per kernel)
    if args.mode == "training-logs":
        best = _best_eval(run_dir)
        per_kernel = [{"kernel": r["benchmark"], "category": _category(r["benchmark"]), "greedy": r}
                      for r in best["greedy_results"]
                      if r["benchmark"] not in EXCLUDED_KERNELS]
        _write_single_outputs(out_dir, args, run_dir, train_config, best, per_kernel)
        logging.info(f"Done. Pulled best eval #{best['eval_count']} (ts={best['timestep']}, "
                     f"mean speedup={np.mean([r['greedy']['speedup'] for r in per_kernel]):.2f}x) "
                     f"-> {out_dir}")
        return

    # ── execution mode: load best_model and re-run live ──
    model_path = run_dir / "best_model" / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(
            f"best_model.zip not found at {model_path}. The run may not have produced "
            f"a best model yet (no eval beat -inf), or training is incomplete."
        )
    logging.info(f"Best model: {model_path}")

    # Rebuild the env exactly as trained (spaces must match the saved policy),
    # then force the eval split and apply the eval-time executor choice.
    env_config = {k: train_config[k] for k in ENV_CONFIG_KEYS if k in train_config}
    env_config.update({
        "benchmarks_split": "eval",
        "executor_type": args.executor_type,
        "verbose": args.verbose,
    })

    if args.executor_type == "dask":
        from llm_action.src.execution.dask_executor import init_shared_client
        init_shared_client(args.dask_nodes)
        logging.info(f"Dask cluster ready with {args.dask_nodes} worker(s)")

    env = ActionMasker(MLIROptEnv(config=env_config), lambda e: e.action_masks())
    model = MaskablePPO.load(str(model_path), env=env)

    try:
        benchmarks = env.unwrapped.benchmarks
        n_benchmarks = len(benchmarks)
        logging.info(f"Evaluating {n_benchmarks} eval benchmark(s) "
                     f"(greedy + {args.eval_sample_runs} sampling run(s) each)")

        per_kernel = []        # rows for per_kernel.csv / results.json
        for idx in range(n_benchmarks):
            if benchmarks[idx].name in EXCLUDED_KERNELS:
                logging.info(f"[{idx + 1}/{n_benchmarks}] {benchmarks[idx].name}: excluded, skipping")
                continue
            greedy_runs = [run_episode(model, env, idx, deterministic=True)
                           for _ in range(args.eval_greedy_runs)]
            stats = _kernel_stats(greedy_runs)
            benchmark = greedy_runs[0]["benchmark"]

            samples = []
            if args.enable_sampling:
                samples = [run_episode(model, env, idx, deterministic=False)
                           for _ in range(args.eval_sample_runs)]

            per_kernel.append({
                "kernel": benchmark,
                "category": _category(benchmark),
                "stats": stats,
                "greedy_runs": greedy_runs,
                "samples": samples,
            })
            if args.verbose:
                sp = stats["speedup"]
                logging.info(
                    f"[{idx + 1}/{n_benchmarks}] {benchmark}: "
                    f"speedup median={sp['median']:.3f}x  [{sp['min']:.3f}, {sp['max']:.3f}] "
                    f"(n={args.eval_greedy_runs})"
                )
    finally:
        env.close()
        if args.executor_type == "dask":
            from llm_action.src.execution.dask_executor import close_shared_client
            close_shared_client()

    _write_outputs(out_dir, args, run_dir, model_path, train_config, per_kernel)
    logging.info(f"Done. Evaluation written to {out_dir}")

def _write_single_outputs(out_dir, args, run_dir, train_config, best, per_kernel):
    """training-logs mode: single value per kernel (the best model's greedy eval)."""
    # ── per_kernel.csv (single value per metric) ──
    pk_fields = ["kernel", "category", "speedup", "exec_time_ms", "torch_time_ms", "speedup_to_torch"]
    with open(out_dir / "per_kernel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pk_fields)
        w.writeheader()
        for r in per_kernel:
            g = r["greedy"]
            w.writerow({
                "kernel": r["kernel"], "category": r["category"],
                "speedup": f"{g['speedup']:.6f}",
                "exec_time_ms": f"{g['opt_time_ms']:.6f}",
                "torch_time_ms": f"{g['torch_time_ms']:.6f}",
                "speedup_to_torch": f"{g['speedup_to_torch']:.6f}",
            })

    # ── summary.csv (per-category groups + OVERALL) ──
    groups = defaultdict(list)
    for r in per_kernel:
        groups[r["category"]].append(r)

    def _summary_row(group, rows):
        sp = [r["greedy"]["speedup"] for r in rows]
        tsp = [r["greedy"]["speedup_to_torch"] for r in rows]
        opt = [r["greedy"]["opt_time_ms"] for r in rows if r["greedy"]["opt_time_ms"] > 0]
        return {
            "group": group, "n_kernels": len(rows),
            "mean_speedup": float(np.mean(sp)), "geomean_speedup": _geomean(sp),
            "mean_exec_time_ms": float(np.mean(opt)) if opt else 0.0,
            "mean_speedup_to_torch": float(np.mean(tsp)), "geomean_speedup_to_torch": _geomean(tsp),
        }

    sm_fields = ["group", "n_kernels", "mean_speedup", "geomean_speedup", "mean_exec_time_ms",
                 "mean_speedup_to_torch", "geomean_speedup_to_torch"]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sm_fields)
        w.writeheader()
        for group in sorted(groups):
            w.writerow(_summary_row(group, groups[group]))
        if per_kernel:
            w.writerow(_summary_row("OVERALL", per_kernel))

    # ── results.json ──
    payload = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "eval_source": "training-logs",
        "best_eval_count": best["eval_count"],
        "best_timestep": best["timestep"],
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "n_benchmarks": len(per_kernel),
        "train_config": train_config,
        "greedy_results": [r["greedy"] for r in per_kernel],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

def _write_outputs(out_dir, args, run_dir, model_path, train_config, per_kernel):
    # CSVs hold the greedy 9-run 5-number summary only (matching the old mlir_rl/
    # schema); stochastic sampling, if enabled, is recorded in results.json only.

    # ── per_kernel.csv: per metric, 5-number summary as actual order statistics ──
    pk_fields = ["kernel", "category"]
    for metric in PER_KERNEL_METRICS:
        pk_fields += [f"{metric}_{s}" for s in STAT_KEYS]
    with open(out_dir / "per_kernel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=pk_fields)
        w.writeheader()
        for r in per_kernel:
            row = {"kernel": r["kernel"], "category": r["category"]}
            for metric in PER_KERNEL_METRICS:
                for s in STAT_KEYS:
                    row[f"{metric}_{s}"] = f"{r['stats'][metric][s]:.6f}"
            w.writerow(row)

    # ── summary.csv: per-kernel stats aggregated across kernels (per group + OVERALL) ──
    groups = defaultdict(list)
    for r in per_kernel:
        groups[r["category"]].append(r)

    def _aggregate(rows, metric, stat, agg):
        vals = [r["stats"][metric][stat] for r in rows]
        return _geomean(vals) if agg == "geomean" else float(np.mean(vals))

    def _summary_row(group, rows):
        row = {"group": group, "n_kernels": len(rows)}
        for metric in PER_KERNEL_METRICS:
            for agg in SUMMARY_AGGS[metric]:
                for s in STAT_KEYS:
                    row[f"{agg}_{metric}_{s}"] = _aggregate(rows, metric, s, agg)
        return row

    sm_fields = ["group", "n_kernels"]
    for metric in PER_KERNEL_METRICS:
        for agg in SUMMARY_AGGS[metric]:
            sm_fields += [f"{agg}_{metric}_{s}" for s in STAT_KEYS]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sm_fields)
        w.writeheader()
        for group in sorted(groups):
            w.writerow(_summary_row(group, groups[group]))
        if per_kernel:
            w.writerow(_summary_row("OVERALL", per_kernel))

    # ── results.json (full record for the thesis) ──
    def _result_entry(r):
        entry = {
            "kernel": r["kernel"],
            "category": r["category"],
            "stats": r["stats"],
            "greedy_runs": r["greedy_runs"],
        }
        if args.enable_sampling:
            entry["samples"] = r["samples"]
        return entry

    payload = {
        "run_name": args.run_name,
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "executor_type": args.executor_type,
        "eval_greedy_runs": args.eval_greedy_runs,
        "sampling_enabled": args.enable_sampling,
        "eval_sample_runs": args.eval_sample_runs if args.enable_sampling else 0,
        "n_benchmarks": len(per_kernel),
        "train_config": train_config,
        "results": [_result_entry(r) for r in per_kernel],
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2))

if __name__ == "__main__":
    main()
