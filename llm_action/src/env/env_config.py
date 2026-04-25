from dataclasses import dataclass

from llm_action.src.config import (
    MAX_STEPS, L, LS, LSD, SLURM_TIMEOUT,
)

@dataclass
class EnvConfig:
    action_version: str = "v10"
    param_mode: str = "multidiscrete"  # "multidiscrete", "two_policy", "llm"
    max_steps: int = MAX_STEPS
    benchmarks_name: str = "matmul"
    executor_type: str = "dask"  # "slurm", "dask", "local"
    slurm_timeout: int = SLURM_TIMEOUT
    dask_timeout: int = 30
    parametrizer_retries: int = 2
    max_num_loops: int = L
    max_num_stores_loads: int = LS
    max_num_load_store_dim: int = LSD
    intermediate_reward: float = 0.0
    failed_transform_penalty: float = -1.0
    failed_exec_penalty: float = -5.0
    no_action_penalty: float = -0.1
    reward_scale: str = "log"
    reward_mode: str = "final"  # "final", "intermediate", "schedule"
    reward_baseline: str = "mlir"  # "mlir" or "torch"
    max_speedup_cap: float = 1000.0
    history_mode: str = "success-encoding"  # "include-all", "ignore-failed", "success-encoding"
    unique_actions: bool = True
    verbose: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid})
