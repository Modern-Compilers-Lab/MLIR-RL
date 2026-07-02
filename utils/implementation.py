from __future__ import annotations
import importlib
import json
import os
import re
from pathlib import Path
from typing import Optional

DEFAULT_IMPLEMENTATION = "rl_autoschedular_v0"

# Preserve legacy naming for already-produced results.
LEGACY_IMPLEMENTATION_META = {
    "rl_autoschedular_v0": {
        "agent_dir": "old_agent",
        "base_prefix": "old",
        "display_name": "Baseline RL",
    },
    "new_rl_autoschedular": {
        "agent_dir": "new_agent",
        "base_prefix": "new",
        "display_name": "New RL",
    },
}


def _load_impl_from_config(config_path: str) -> Optional[str]:
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    impl = config_data.get("implementation")
    if not isinstance(impl, str):
        return None

    impl = impl.strip()
    return impl or None


def get_autoschedular_impl(default: str = DEFAULT_IMPLEMENTATION, config_path: Optional[str] = None) -> str:
    """Return selected autoscheduler implementation.

    Resolution order:
      1) AUTOSCHEDULER_IMPL env var
      2) explicit config_path (if provided)
      3) CONFIG_FILE_PATH -> config["implementation"]
      3) provided default
    """
    env_impl = os.getenv("AUTOSCHEDULER_IMPL", "").strip()
    if env_impl:
        return env_impl

    candidate_paths = []
    if config_path:
        candidate_paths.append(config_path)
    env_config_path = os.getenv("CONFIG_FILE_PATH", "").strip()
    if env_config_path:
        candidate_paths.append(env_config_path)

    for candidate_path in candidate_paths:
        cfg_impl = _load_impl_from_config(candidate_path)
        if cfg_impl:
            return cfg_impl

    return default


def _implementation_token(implementation: str) -> str:
    if implementation in LEGACY_IMPLEMENTATION_META:
        return LEGACY_IMPLEMENTATION_META[implementation]["base_prefix"]

    # Canonical names for versioned agents:
    #   rl_autoschedular_v1 -> v1
    #   rl_autoschedular_v2 -> v2
    #   rl_autoschedular_v2_5 -> v2_5
    version_match = re.fullmatch(r"rl_autoschedular_v([\d_]+)", implementation)
    if version_match:
        return f"v{version_match.group(1)}"

    # Fallback for custom names
    token = re.sub(r"[^a-zA-Z0-9]+", "_", implementation).strip("_").lower()
    return token or "custom"


def get_implementation_meta(implementation: Optional[str] = None) -> dict[str, str]:
    """Return naming metadata for implementation specific artifacts."""
    impl = implementation or get_autoschedular_impl()

    if impl in LEGACY_IMPLEMENTATION_META:
        return {
            "implementation": impl,
            "agent_dir": LEGACY_IMPLEMENTATION_META[impl]["agent_dir"],
            "base_prefix": LEGACY_IMPLEMENTATION_META[impl]["base_prefix"],
            "display_name": LEGACY_IMPLEMENTATION_META[impl]["display_name"],
        }

    token = _implementation_token(impl)
    display_name = token.upper() + " RL" if token.startswith("v") else impl
    return {
        "implementation": impl,
        "agent_dir": f"{token}_agent",
        "base_prefix": token,
        "display_name": display_name,
    }


def get_agent_subdir(implementation: Optional[str] = None) -> str:
    """Return canonical agent subdir name under results/<experiment>/ for an implementation."""
    return get_implementation_meta(implementation)["agent_dir"]


def get_base_prefix(implementation: Optional[str] = None) -> str:
    """Return canonical baseline filename prefix for an implementation."""
    return get_implementation_meta(implementation)["base_prefix"]


def get_agent_runs_root(results_dir: str, implementation: Optional[str] = None) -> Path:
    """Return results/<experiment>/ directly — no impl subdir nesting."""
    return Path(results_dir)


def get_base_file_path(results_dir: str, implementation: Optional[str] = None) -> Path:
    """Return results/<experiment>/exec_times/<prefix>_base.json.

    If implementation is None, returns the dataset-level generic path
    (exec_times/base.json). This is the preferred path for shared baselines.
    """
    if implementation is None:
        return Path(results_dir) / "exec_times" / "base.json"
    prefix = get_base_prefix(implementation)
    return Path(results_dir) / "exec_times" / f"{prefix}_base.json"


def get_split_file_path(results_dir: str, implementation: str, is_training: bool) -> Path:
    """Return the path to train or eval split JSON.

    First checks for the implementation-specific split file
    (e.g. exec_times/old_base_train.json), then falls back to the
    dataset-level generic split (exec_times/base_train.json).
    """
    prefix = get_base_prefix(implementation)
    suffix = "train" if is_training else "eval"
    specific = Path(results_dir) / "exec_times" / f"{prefix}_base_{suffix}.json"
    generic = Path(results_dir) / "exec_times" / f"base_{suffix}.json"
    return specific if specific.exists() else generic


def get_base_eval_files(results_dir: str, impl_tokens: list[str]) -> dict[str, dict]:
    """Load baseline eval JSONs for multiple implementations.

    For each implementation token, tries the implementation-specific file first,
    then falls back to the generic base_eval.json (shared across implementations).
    Returns a dict mapping token -> parsed JSON dict.
    """
    import json
    baselines: dict[str, dict] = {}
    generic_path = Path(results_dir) / "exec_times" / "base_eval.json"
    generic_data = None

    for token in impl_tokens:
        specific_path = Path(results_dir) / "exec_times" / f"{token}_base_eval.json"
        if specific_path.exists():
            try:
                with open(specific_path) as f:
                    baselines[token] = json.load(f)
                continue
            except (OSError, json.JSONDecodeError):
                pass
        if generic_data is None and generic_path.exists():
            try:
                with open(generic_path) as f:
                    generic_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        if generic_data is not None:
            baselines[token] = generic_data
    return baselines


def import_autoschedular_module(module: str, implementation: Optional[str] = None):
    """Import a module from the selected autoscheduler implementation package."""
    impl = implementation or get_autoschedular_impl()
    target = impl if not module else f"{impl}.{module}"

    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", "")
        if missing not in {impl, target}:
            raise
        raise ModuleNotFoundError(
            f"Could not import autoscheduler implementation '{target}'. "
            "Set AUTOSCHEDULER_IMPL or config['implementation'] to a valid package "
            f"and ensure the package exists (for example, add {impl}/__init__.py)."
        ) from exc
