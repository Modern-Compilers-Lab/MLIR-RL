"""RL-side benchmark loader.

File discovery, family detection, and baseline I/O live in
`llm_action.src.data.benchmarks`. This module wraps that layer with the
RL-specific concerns: triggering MLIR baseline measurement via the executor
and torch baselines via SLURM.
"""

import logging
import re
from dataclasses import dataclass

from llm_action.src.data.benchmarks import (
    Split,
    load_baselines,
    load_benchmark_set,
    save_baselines,
)

logger = logging.getLogger(__name__)


@dataclass
class Benchmark:
    name: str
    code: str
    base_exec_time_ms: float
    torch_exec_time_ms: float = -1.0


_MATMUL_PATTERN = re.compile(r"matmul_(\d+)_(\d+)_(\d+)$")
_CONV2D_PATTERN = re.compile(r"conv_2d_nchw_fchw_" + "_".join([r"(\d+)"] * 9) + r"$")
_ADD_PATTERN = re.compile(r"add_(\d+)_(\d+)_(\d+)_(\d+)$")
_POOLING_NCHW_MAX_PATTERN = re.compile(
    r"pooling_nchw_max_" + "_".join([r"(\d+)"] * 7) + r"$"
)
_RELU_PATTERN = re.compile(r"relu_(\d+(?:_\d+)*)$")

def _parse_op_dims(name: str) -> tuple[str, tuple[int, ...]] | None:
    """Return (torch CLI subcommand, args) parsed from a benchmark filename, or None.

    The returned tuple is exactly what `torch.sh <subcommand> <args...>` expects,
    so callers can forward the args verbatim.

    Supported patterns:
        matmul_M_K_N                                          -> ("matmul", (M, K, N))
        conv_2d_nchw_fchw_N_C_H_W_F_KH_KW_OH_OW               -> ("conv2d", (N, C, H, W, F, KH, KW, OH, OW))
        add_A_B_C_D                                           -> ("add", (A, B, C, D))
        pooling_nchw_max_N_C_H_W_K_OH_OW (square kernel)      -> ("pooling_nchw_max", (N, C, H, W, K, K, OH, OW))
        relu_D1[_D2[_D3...]]                                  -> ("relu", (D1, D2, ...))
    """
    if m := _MATMUL_PATTERN.match(name):
        return ("matmul", tuple(int(x) for x in m.groups()))
    if m := _CONV2D_PATTERN.match(name):
        return ("conv2d", tuple(int(x) for x in m.groups()))
    if m := _ADD_PATTERN.match(name):
        return ("add", tuple(int(x) for x in m.groups()))
    if m := _POOLING_NCHW_MAX_PATTERN.match(name):
        N, C, H, W, K, OH, OW = (int(x) for x in m.groups())
        return ("pooling_nchw_max", (N, C, H, W, K, K, OH, OW))
    if m := _RELU_PATTERN.match(name):
        return ("relu", tuple(int(x) for x in m.group(1).split("_")))
    return None

def measure_torch_baseline(op_type: str, dims: tuple[int, ...], executor) -> float:
    """Measure PyTorch execution time for the given op via the env's executor.

    With executor_type="dask" runs on a Dask worker (no SLURM queue); with
    "slurm" submits torch.sh; with "local" runs in-process. Returns median
    execution time in milliseconds.
    """
    return executor.execute_torch(op_type, dims)

def load_benchmarks(
    name: str = "standard",
    split: Split = "train",
    executor=None,
) -> list[Benchmark]:
    """Load a benchmark set, attaching cached or freshly-measured baselines."""
    instances = load_benchmark_set(name, split=split)
    cache = load_baselines(name, split=split)

    benchmarks: list[Benchmark] = []
    needs_mlir_measurement: list[int] = []
    needs_torch_measurement: list[int] = []

    for inst in instances:
        entry = cache.get(inst.name, {})
        mlir_time = entry.get("mlir", -1.0)
        torch_time = entry.get("torch", -1.0)
        benchmarks.append(Benchmark(inst.name, inst.code, mlir_time, torch_time))
        idx = len(benchmarks) - 1
        if mlir_time < 0:
            needs_mlir_measurement.append(idx)
        if torch_time < 0:
            needs_torch_measurement.append(idx)

    if needs_mlir_measurement and executor is not None:
        logger.info(f"Measuring MLIR baselines for {len(needs_mlir_measurement)} benchmarks...")
        for idx in needs_mlir_measurement:
            b = benchmarks[idx]
            try:
                t, ok = executor.execute(b.code)
                b.base_exec_time_ms = t if ok and t > 0 else 1.0
                logger.info(f"  {b.name}: {b.base_exec_time_ms:.2f} ms")
            except Exception as e:
                logger.warning(f"  {b.name}: MLIR baseline failed: {e}")
                b.base_exec_time_ms = 1.0
            cache.setdefault(b.name, {})["mlir"] = b.base_exec_time_ms
    elif needs_mlir_measurement:
        logger.warning(f"{len(needs_mlir_measurement)} benchmarks need MLIR baseline measurement (deferred to reset)")

    if needs_torch_measurement and executor is not None:
        logger.info(f"Measuring torch baselines for {len(needs_torch_measurement)} benchmarks...")
        for idx in needs_torch_measurement:
            b = benchmarks[idx]
            parsed = _parse_op_dims(b.name)
            if parsed is None:
                logger.warning(f"  {b.name}: cannot parse dimensions for torch baseline, skipping")
                continue
            op_type, dims = parsed
            try:
                t = measure_torch_baseline(op_type, dims, executor)
                b.torch_exec_time_ms = t
                cache.setdefault(b.name, {})["torch"] = t
                logger.info(f"  {b.name}: torch {t:.2f} ms")
            except Exception as e:
                logger.warning(f"  {b.name}: torch baseline failed: {e}")
    elif needs_torch_measurement:
        logger.warning(f"{len(needs_torch_measurement)} benchmarks need torch baseline measurement (no executor configured)")

    save_baselines(name, cache, split=split)

    return benchmarks
