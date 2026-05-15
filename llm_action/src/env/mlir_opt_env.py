import logging
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from llm_action.src.env.action_registry import load_action_registry, ActionRegistry
from llm_action.src.env.benchmarks import Benchmark, load_benchmarks
from llm_action.src.env.env_config import EnvConfig
from llm_action.src.execution.local_executer import LocalExecutor
from llm_action.src.execution.dask_executor import DaskExecutor, get_shared_client
from llm_action.src.execution.slurm_executor import SlurmExecutor
from llm_action.src.env.state_extractor import extract_observation, observation_size, count_loops
from llm_action.src.env.action_space import build_action_space, build_action_masks, compute_blocked_indices
from llm_action.src.config import L, MAX_ACTION_EXECUTIONS

logger = logging.getLogger(__name__)

class MLIROptEnv(gym.Env):
    """MLIR optimization environment with configurable parametrization.

    param_mode:
      "multidiscrete" : single policy outputs [action, slot_0..slot_6]
      "two_policy"    : Discrete action space, internal param model picks params
      "llm"           : Discrete action space, LLM generates params
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = EnvConfig.from_dict(config) if config else EnvConfig()
        self.cfg = cfg
        self.verbose = cfg.verbose
        self.param_mode = cfg.param_mode

        self.registry: ActionRegistry = load_action_registry(cfg.action_version)
        reg = self.registry

        # Action space depends on param_mode
        if self.param_mode == "multidiscrete":
            self.action_space, self._slot_map = build_action_space(reg, cfg.max_num_loops)
        else:
            self.action_space = spaces.Discrete(reg.total_actions)
            self._slot_map = {}

        obs_sz = observation_size(reg.total_actions, cfg.max_steps, cfg.history_mode)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_sz,), dtype=np.float32
        )

        if cfg.executor_type == "dask":
            self.executor = DaskExecutor(client=get_shared_client(), timeout=cfg.dask_timeout)
        elif cfg.executor_type == "local":
            self.executor = LocalExecutor()
        else:
            self.executor = SlurmExecutor(timeout=cfg.slurm_timeout)
        self.benchmarks = load_benchmarks(
            name=cfg.benchmarks_name,
            split=cfg.benchmarks_split,
            executor=self.executor,
        )
        assert self.benchmarks, "No benchmarks loaded"

        # Parametrizer: only created for LLM mode
        self._llm_parametrizer = None
        if self.param_mode == "llm":
            from llm_action.src.agents.parametrizer import ParametrizerAgent
            self._llm_parametrizer = ParametrizerAgent()

        # Two-policy mode: param model set externally via set_param_model()
        self._param_model = None

        self._benchmark: Benchmark | None = None
        self._current_code = ""
        self._action_history: list[tuple[str, dict]] = []
        self._action_indices: list[tuple[int, bool]] = []
        self._used_action_counts: dict[int, int] = {}
        self._step_count = 0
        self._episode_count = 0
        self._last_exec_time_ms: float = -1.0
        self._n_loops = L

    @property
    def param_observation_size(self) -> int:
        """Input size for the two_policy param model: obs_size + total_actions (one-hot)."""
        return observation_size(self.registry.total_actions, self.cfg.max_steps) + self.registry.total_actions

    def set_param_model(self, model):
        """Set the parameter policy for two_policy mode. Called from training script."""
        self._param_model = model

    # Gymnasium API

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._episode_count += 1
        benchmark_idx = (options or {}).get("benchmark_idx")
        if benchmark_idx is not None:
            self._benchmark = self.benchmarks[benchmark_idx]
        else:
            self._benchmark = self.benchmarks[self.np_random.integers(0, len(self.benchmarks))]
        if self._benchmark.base_exec_time_ms < 0:
            self._measure_baseline(self._benchmark)
        self._current_code = self._benchmark.code
        self._action_history = []
        self._action_indices: list[tuple[int, bool]] = []
        self._used_action_counts: dict[int, int] = {}
        self._step_count = 0
        self._last_exec_time_ms = -1.0

        self._log(f"\n{'='*60}")
        self._log(f"[EP {self._episode_count}] RESET | benchmark={self._benchmark.name} | "
                   f"base={self._benchmark.base_exec_time_ms:.2f}ms")

        self._n_loops = count_loops(self._current_code)
        obs = self._obs()

        return obs, {"benchmark": self._benchmark.name}

    def action_masks(self) -> np.ndarray:
        """Boolean mask over the action space for MaskablePPO.

        True  = action available.
        False = action masked out (policy assigns −∞ logit).

        Per-action `unique_execution` (ActionBase class attribute, default True)
        decides whether a successfully-applied action is masked for the rest of
        the episode. The done action is always unmasked so the episode can
        terminate.
        """
        blocked = (
            compute_blocked_indices(self.registry, set(self._used_action_counts.keys()))
            if self.cfg.enable_dependency_masking
            else frozenset()
        )

        if self.param_mode == "multidiscrete":
            return build_action_masks(
                self.registry, self._slot_map,
                n_loops=self._n_loops,
                max_n_loops=self.cfg.max_num_loops,
                used_action_counts=self._used_action_counts,
                blocked_by_dependency=blocked,
            )

        reg = self.registry
        action_mask = np.ones(reg.total_actions, dtype=bool)
        for idx, count in self._used_action_counts.items():
            cap = 1 if reg.action_classes[idx].unique_execution else MAX_ACTION_EXECUTIONS
            if count >= cap:
                action_mask[idx] = False
        for idx in blocked:
            action_mask[idx] = False
        action_mask[reg.done_idx] = True
        return action_mask

    def step(self, action):
        self._step_count += 1
        reg = self.registry

        # Unpack action + param slots depending on mode
        action_idx, param_slots = self._unpack_action(action)

        if action_idx == reg.done_idx:
            reward, opt_t = self._terminal_reward()
            self._action_indices.append((action_idx, True))
            self._log(f"  step {self._step_count}: DONE | reward={reward:.4f} | "
                       f"opt={opt_t:.2f}ms | history={[h[0] for h in self._action_history]}")
            self._log_episode_summary(reward, opt_t)
            return self._obs(), reward, True, False, self._info(reward, opt_time_ms=opt_t)

        _used = self._used_action_counts.get(action_idx, 0)
        _cap = 1 if reg.action_classes[action_idx].unique_execution else MAX_ACTION_EXECUTIONS
        if _used >= _cap:
            # Unreachable during MaskablePPO training; safety net for unmasked inference.
            action_name = reg.action_classes[action_idx].__name__
            self._log(f"  step {self._step_count}: {action_name} | FAIL (cap reached: {_used}/{_cap}) [mask bypass]")
            return self._fail_step(action_idx)

        action_class = reg.action_classes[action_idx]
        action_name = action_class.__name__

        # Get params from the appropriate source
        params = self._resolve_params(action_class, param_slots)

        if params is None:
            self._log(f"  step {self._step_count}: {action_name} | FAIL (params=None)")
            return self._fail_step(action_idx)

        if not action_class.precondition(self._current_code, params):
            has_tag = 'tag = "operation_0"' in self._current_code
            self._log(f"  step {self._step_count}: {action_name}({params}) | FAIL (precondition) | has_tag={has_tag}")
            return self._fail_step(action_idx)

        try:
            pre = action_class.preprocess(self._current_code, params)
            new_code = action_class.implement(pre, params)
            if not action_class.postcondition(self._current_code, new_code, params):
                self._log(f"  step {self._step_count}: {action_name}({params}) | FAIL (postcondition)")
                return self._fail_step(action_idx)
        except Exception as e:
            self._log(f"  step {self._step_count}: {action_name}({params}) | FAIL ({e})")
            return self._fail_step(action_idx)

        self._current_code = new_code
        self._action_history.append((action_name, params))
        self._action_indices.append((action_idx, True))
        self._used_action_counts[action_idx] = self._used_action_counts.get(action_idx, 0) + 1

        # Update loop count in case the action changed the op structure
        # (e.g., Image2Col converts 7-loop conv2d to 4-loop generic)
        self._n_loops = count_loops(self._current_code)

        # If the tag was consumed (e.g., Vectorization), auto-terminate
        tag_gone = 'tag = "operation_0"' not in self._current_code
        if tag_gone:
            reward, opt_t = self._terminal_reward()
            self._log(f"  step {self._step_count}: {action_name}({params}) | OK (terminal — tag consumed) | reward={reward:.4f}")
            self._log_episode_summary(reward, opt_t)
            return self._obs(), reward, True, False, self._info(reward, opt_time_ms=opt_t)

        truncated = self._step_count >= self.cfg.max_steps
        opt_t = -1.0

        if truncated:
            reward, opt_t = self._terminal_reward()
        elif self.cfg.reward_mode != "final":
            reward = self._step_reward()
        else:
            reward = self.cfg.intermediate_reward

        self._log(f"  step {self._step_count}: {action_name}({params}) | OK | reward={reward:.4f}")
        if truncated:
            self._log_episode_summary(reward, opt_t)

        return self._obs(), reward, False, truncated, self._info(reward, opt_time_ms=opt_t)

    # Action/param unpacking

    def _unpack_action(self, action) -> tuple[int, list[int] | None]:
        if self.param_mode == "multidiscrete":
            arr = np.asarray(action, dtype=int)
            action_idx = int(arr[0]) % self.registry.total_actions
            start, end = self._slot_map.get(action_idx, (0, 0))
            param_slots = arr[1 + start:1 + end].tolist() if end > start else []
            return action_idx, param_slots
        else:
            return int(action) % self.registry.total_actions, None

    def _resolve_params(self, action_class, param_slots: list[int] | None) -> dict | None:
        n = self._n_loops

        if self.param_mode == "multidiscrete":
            return action_class.decode_params(param_slots or [], n_loops=n)

        if self.param_mode == "two_policy" and self._param_model is not None:
            param_obs = self._build_param_obs(action_class)
            raw_slots, _ = self._param_model.predict(param_obs, deterministic=False)
            return action_class.decode_params(raw_slots.tolist(), n_loops=n)

        if self.param_mode == "llm" and self._llm_parametrizer is not None:
            for _ in range(self.cfg.parametrizer_retries + 1):
                try:
                    return self._llm_parametrizer.parametrize(
                        self._current_code, action_class, self._action_history
                    )
                except Exception:
                    pass
            return None

        return action_class.decode_params([0] * action_class.params_size(), n_loops=n)

    def _build_param_obs(self, action_class) -> np.ndarray:
        """Observation for the param policy: base obs + one-hot action."""
        obs = self._obs()
        action_idx = self.registry.name_to_idx.get(action_class.__name__, 0)
        one_hot = np.zeros(self.registry.total_actions, dtype=np.float32)
        one_hot[action_idx] = 1.0
        return np.concatenate([obs, one_hot])

    # Shared helpers

    def _fail_step(self, action_idx: int):
        self._action_indices.append((action_idx, False))
        truncated = self._step_count >= self.cfg.max_steps
        reward = self.cfg.failed_transform_penalty
        opt_t = -1.0
        if truncated:
            reward, opt_t = self._terminal_reward()
            self._log_episode_summary(reward, opt_t)
        return self._obs(), reward, False, truncated, self._info(reward, failed=True, opt_time_ms=opt_t)

    def _log_episode_summary(self, reward: float, opt_t: float):
        base = self._benchmark.base_exec_time_ms
        actions = [h[0] for h in self._action_history]
        speedup = base / opt_t if opt_t > 0 else 0
        self._log(f"  ── EP {self._episode_count} SUMMARY ──")
        self._log(f"  benchmark: {self._benchmark.name}")
        self._log(f"  actions ({len(actions)}): {' → '.join(actions) if actions else '(none)'}")
        self._log(f"  base={base:.2f}ms | opt={opt_t:.2f}ms | speedup={speedup:.2f}x | reward={reward:.4f}")
        self._log(f"{'='*60}\n")

    def _measure_baseline(self, b: Benchmark):
        try:
            t, ok = self.executor.execute(b.code)
            b.base_exec_time_ms = t if ok and t > 0 else 1.0
        except Exception as e:
            logger.warning(f"Baseline error for {b.name}: {e}")
            b.base_exec_time_ms = 1.0

    def _get_reward_baseline(self) -> float:
        """Return the baseline time to compute speedup ratios against."""
        if self.cfg.reward_baseline == "torch":
            torch_t = self._benchmark.torch_exec_time_ms
            if torch_t > 0:
                return torch_t
            logger.warning(f"Torch baseline unavailable for {self._benchmark.name}, falling back to MLIR baseline")
        return self._benchmark.base_exec_time_ms

    def _step_reward(self) -> float:
        """Compute per-step reward for intermediate/schedule modes."""
        try:
            t, ok = self.executor.execute(self._current_code)
        except Exception:
            return self.cfg.failed_exec_penalty
        if not ok or t <= 0:
            return self.cfg.failed_exec_penalty
        base = self._get_reward_baseline()
        if base <= 0:
            return 0.0

        if self.cfg.reward_mode == "intermediate":
            prev = self._last_exec_time_ms if self._last_exec_time_ms > 0 else base
            ratio = prev / t
        else:  # schedule
            ratio = base / t

        if ratio > self.cfg.max_speedup_cap:
            return self.cfg.failed_exec_penalty

        self._last_exec_time_ms = t
        return self._compute_reward(ratio)

    def _terminal_reward(self) -> tuple[float, float]:
        if not self._action_history:
            return self.cfg.no_action_penalty, -1.0
        try:
            t, ok = self.executor.execute(self._current_code)
        except Exception:
            return self.cfg.failed_exec_penalty, -1.0
        if not ok or t <= 0:
            return self.cfg.failed_exec_penalty, t
        base = self._get_reward_baseline()
        if base <= 0:
            return 0.0, t

        ratio = base / t
        if ratio > self.cfg.max_speedup_cap:
            logger.warning(f"Degenerate speedup {ratio:.1f}x for {self._benchmark.name} "
                           f"(base={base:.2f}ms, opt={t:.4f}ms) — penalizing")
            return self.cfg.failed_exec_penalty, t

        return self._compute_reward(ratio), t

    def _compute_reward(self, ratio: float) -> float:
        match self.cfg.reward_scale:
            case "raw":
                return ratio
            case "delta":
                return 1.0 - (1.0 / ratio)
            case _:
                return math.log(ratio)

    def _obs(self) -> np.ndarray:
        return extract_observation(
            self._current_code, self._action_indices, self._step_count,
            self.registry.total_actions, self.cfg.max_steps, self.cfg.history_mode
        )

    def _info(self, reward: float, failed: bool = False, opt_time_ms: float = -1) -> dict:
        return {
            "benchmark": self._benchmark.name,
            "step": self._step_count,
            "action_history": [h[0] for h in self._action_history],
            "action_outcomes": [
                (self.registry.action_classes[idx].__name__, ok)
                for idx, ok in self._action_indices
                if idx < len(self.registry.action_classes)
            ],
            "reward": reward,
            "failed": failed,
            "base_time_ms": self._benchmark.base_exec_time_ms if self._benchmark else -1,
            "torch_time_ms": self._benchmark.torch_exec_time_ms if self._benchmark else -1,
            "opt_time_ms": opt_time_ms,
        }
