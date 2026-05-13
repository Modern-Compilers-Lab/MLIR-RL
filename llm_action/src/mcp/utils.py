from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

from llm_action.src.config import (
    PROJECT_ROOT,
    MLIR_SCRIPT,
    MLIR_TMP_DIR,
    MLIR_SLURM_LOG_DIR,
    TORCH_SCRIPT,
    TORCH_SLURM_LOG_DIR,
    SLURM_TIMEOUT,
    SLURM_POLL_INTERVAL,
)


def _submit_sbatch(cmd: list[str]) -> str:
    """Run an sbatch command and return the parsed job ID."""
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID: {result.stdout.strip()}")
    return match.group(1)


def _wait_for_job(job_id: str, timeout: int = SLURM_TIMEOUT) -> None:
    """Poll squeue until the job is gone, or raise TimeoutError after `timeout` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sq = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True, text=True,
        )
        if not sq.stdout.strip():
            return
        time.sleep(SLURM_POLL_INTERVAL)
    raise TimeoutError(f"SLURM job {job_id} did not finish within {timeout}s")


def run_torch_sbatch(args: list[str]) -> float:
    """Submit `torch.sh args...` via sbatch, wait, and return the parsed median ms.

    The torch CLI prints a single float on the last stdout line; this reads it
    from the SLURM .out log.
    """
    job_id = _submit_sbatch(["sbatch", str(TORCH_SCRIPT), *args])
    _wait_for_job(job_id)

    log_path = TORCH_SLURM_LOG_DIR / f"{job_id}.out"
    error_path = TORCH_SLURM_LOG_DIR / f"{job_id}.err"
    output = log_path.read_text().strip() if log_path.exists() else ""
    error_output = error_path.read_text().strip() if error_path.exists() else ""

    if not log_path.exists():
        raise FileNotFoundError(
            f"SLURM log not found: {log_path}\nstderr:\n{error_output}"
        )

    try:
        return float(output.splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(
            f"torch job {job_id} did not produce a parseable result.\n"
            f"stdout:\n{output}\nstderr:\n{error_output}"
        )


def torch_matmul_by_shape(M: int, K: int, N: int) -> float:
    return run_torch_sbatch(["matmul", str(M), str(K), str(N)])


def torch_conv2d_by_shape(
    N: int, C: int, H: int, W: int,
    F: int, KH: int, KW: int, OH: int, OW: int,
) -> float:
    return run_torch_sbatch([
        "conv2d",
        str(N), str(C), str(H), str(W),
        str(F), str(KH), str(KW), str(OH), str(OW),
    ])


def torch_add_by_shape(A: int, B: int, C: int, D: int) -> float:
    return run_torch_sbatch(["add", str(A), str(B), str(C), str(D)])


def torch_pooling_nchw_max_by_shape(
    N: int, C: int, H: int, W: int,
    KH: int, KW: int, OH: int, OW: int,
) -> float:
    return run_torch_sbatch([
        "pooling_nchw_max",
        str(N), str(C), str(H), str(W),
        str(KH), str(KW), str(OH), str(OW),
    ])


def torch_relu_by_shape(shape: list[int]) -> float:
    if not shape:
        raise ValueError("relu shape must have at least one dimension")
    return run_torch_sbatch(["relu", *(str(d) for d in shape)])


def run_mlir_sbatch(
    code: str,
    bufferization_lowering_v_transform_code: Optional[str] = None,
    pass_pipeline: Optional[list[str]] = None,
) -> tuple[float, bool]:
    """Submit an MLIR job via mlir.sh and return (median execution time ms, success).

    Writes `code` (and optionally `bufferization_lowering_v_transform_code`) to
    temp files in MLIR_TMP_DIR, passes them to the SLURM script, waits for the
    job, and parses the JSON record on the last stdout line of the .out log.
    Temp files are always cleaned up regardless of outcome.
    """
    MLIR_TMP_DIR.mkdir(parents=True, exist_ok=True)

    code_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", dir=MLIR_TMP_DIR, delete=False
    )
    code_file.write(code)
    code_file.close()

    transform_file = None
    job_id: Optional[str] = None
    try:
        cmd: list[str] = ["sbatch", str(MLIR_SCRIPT), code_file.name]

        if bufferization_lowering_v_transform_code is not None:
            transform_file = tempfile.NamedTemporaryFile(
                mode="w", suffix=".mlir", dir=MLIR_TMP_DIR, delete=False
            )
            transform_file.write(bufferization_lowering_v_transform_code)
            transform_file.close()
            cmd.extend(["--transform-file", transform_file.name])

        if pass_pipeline is not None:
            cmd.extend(["--pass-pipeline", *pass_pipeline])

        job_id = _submit_sbatch(cmd)
        _wait_for_job(job_id)
    finally:
        try:
            Path(code_file.name).unlink(missing_ok=True)
            if transform_file is not None:
                Path(transform_file.name).unlink(missing_ok=True)
        except OSError:
            pass

    assert job_id is not None  # _submit_sbatch returned without raising

    log_path = MLIR_SLURM_LOG_DIR / f"{job_id}.out"
    if not log_path.exists():
        raise FileNotFoundError(f"SLURM log not found: {log_path}")
    output = log_path.read_text().strip()

    error_path = MLIR_SLURM_LOG_DIR / f"{job_id}.err"
    error_output = error_path.read_text().strip() if error_path.exists() else ""

    try:
        result_data = json.loads(output.splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError(
            f"Could not parse result from job {job_id} output:\n{output}, error:\n{error_output}"
        )

    if "error" in result_data:
        raise RuntimeError(
            f"MLIR execution failed (job {job_id}): output:\n{output}, error:\n{error_output}"
        )

    return result_data["execution_time_ms"], result_data["success"]


def compute_speedup(
    mlir_base_execution_time: float,
    mlir_optimized_execution_time: float,
    torch_execution_time: Optional[float] = None,
) -> dict[str, float]:
    """Speedup of optimized MLIR vs base, and (optionally) vs PyTorch baseline."""
    return {
        "speedup": mlir_base_execution_time / mlir_optimized_execution_time,
        "speedup_to_torch": (
            torch_execution_time / mlir_optimized_execution_time
            if torch_execution_time
            else -1
        ),
    }
