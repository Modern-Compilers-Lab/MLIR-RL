import json
import logging
import re
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass

from llm_action.src.config import DATA_DIR, TORCH_SCRIPT, TORCH_SLURM_LOG_DIR

logger = logging.getLogger(__name__)


@dataclass
class Benchmark:
    name: str
    code: str
    base_exec_time_ms: float
    torch_exec_time_ms: float = -1.0


_CONV2D_PATTERN = re.compile(r"conv_2d_nchw_fchw_" + "_".join([r"(\d+)"] * 9) + r"$")
_MATMUL_PATTERN = re.compile(r"matmul_(\d+)_(\d+)_(\d+)$")


def _parse_op_dims(name: str) -> tuple[str, tuple[int, ...]] | None:
    """Return (op_type, dims) parsed from a benchmark filename, or None.

    Supported patterns:
        matmul_M_K_N                                              -> ("matmul", (M, K, N))
        conv_2d_nchw_fchw_N_C_H_W_F_KH_KW_OH_OW                   -> ("conv2d", (N, C, H, W, F, KH, KW, OH, OW))
    """
    if m := _MATMUL_PATTERN.match(name):
        return ("matmul", tuple(int(x) for x in m.groups()))
    if m := _CONV2D_PATTERN.match(name):
        return ("conv2d", tuple(int(x) for x in m.groups()))
    return None


def measure_torch_baseline(op_type: str, dims: tuple[int, ...], timeout: int = 300) -> float:
    """Submit a SLURM job to measure PyTorch execution time for the given op.

    Args:
        op_type: "matmul" or "conv2d" — selects the torch_execution.py subcommand.
        dims: positional dimensions to pass after the subcommand.

    Returns median execution time in milliseconds.
    """
    TORCH_SLURM_LOG_DIR.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        ["sbatch", str(TORCH_SCRIPT), op_type, *map(str, dims)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID: {result.stdout.strip()}")
    job_id = match.group(1)

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sq = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True,
        )
        if not sq.stdout.strip():
            break
        time.sleep(2)
    else:
        raise TimeoutError(f"Torch SLURM job {job_id} did not finish within {timeout}s")

    log_path = TORCH_SLURM_LOG_DIR / f"{job_id}.out"
    if not log_path.exists():
        raise FileNotFoundError(f"SLURM log not found: {log_path}")
    output = log_path.read_text().strip()

    try:
        return float(output.splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(f"Could not parse torch time from job {job_id}:\n{output}")


def _load_cache(cache_path: Path) -> dict[str, dict[str, float]]:
    """Load baselines.json. Format: {"benchmark_name": {"mlir": float, "torch": float}}"""
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache_path: Path, unified: dict[str, dict[str, float]]):
    try:
        cache_path.write_text(json.dumps(unified, indent=2, sort_keys=True))
    except OSError:
        pass


def load_benchmarks(benchmarks_dir=DATA_DIR, name="matmul", executor=None) -> list[Benchmark]:
    bdir = Path(benchmarks_dir) / name
    if not bdir.exists():
        raise FileNotFoundError(f"Benchmarks directory not found: {bdir}")

    mlir_files = sorted(bdir.glob("*.mlir"))
    if not mlir_files:
        raise FileNotFoundError(f"No .mlir files in {bdir}")

    cache_path = bdir / "baselines.json"
    cache = _load_cache(cache_path)

    benchmarks = []
    needs_mlir_measurement = []
    needs_torch_measurement = []

    for f in mlir_files:
        bname = f.stem
        code = f.read_text()
        entry = cache.get(bname, {})
        mlir_time = entry.get("mlir", -1.0)
        torch_time = entry.get("torch", -1.0)
        benchmarks.append(Benchmark(bname, code, mlir_time, torch_time))

        if mlir_time < 0:
            needs_mlir_measurement.append(len(benchmarks) - 1)
        if torch_time < 0:
            needs_torch_measurement.append(len(benchmarks) - 1)

    # Measure MLIR baselines
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

    # Measure torch baselines via SLURM
    if needs_torch_measurement:
        logger.info(f"Measuring torch baselines for {len(needs_torch_measurement)} benchmarks...")
        for idx in needs_torch_measurement:
            b = benchmarks[idx]
            parsed = _parse_op_dims(b.name)
            if parsed is None:
                logger.warning(f"  {b.name}: cannot parse dimensions for torch baseline, skipping")
                continue
            op_type, dims = parsed
            try:
                t = measure_torch_baseline(op_type, dims)
                b.torch_exec_time_ms = t
                cache.setdefault(b.name, {})["torch"] = t
                logger.info(f"  {b.name}: torch {t:.2f} ms")
            except Exception as e:
                logger.warning(f"  {b.name}: torch baseline failed: {e}")

    _save_cache(cache_path, cache)

    return benchmarks
