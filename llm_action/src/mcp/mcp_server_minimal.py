import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from fastmcp import FastMCP
from llm_action.src.utils.transformation import BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE, PASS_PIPELINE
from llm_action.src.config import PROJECT_ROOT, MLIR_SCRIPT, MLIR_TMP_DIR, MLIR_SLURM_LOG_DIR

mcp = FastMCP("mlir-optimization-tools-minimal")

@mcp.tool()
def execute_mlir_code(code: str) -> tuple[float, bool]:
    F"""
    Submits a SLURM job to execute the given MLIR code on a dedicated compute
    node and returns the median execution time.

    Args:
        code (str): The MLIR code to execute.

    Returns:
        tuple[float, bool]: (median execution time in milliseconds, assertion result)
    """
    MLIR_TMP_DIR.mkdir(parents=True, exist_ok=True)

    # Write the MLIR code to a temp file so the SLURM job can read it
    code_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", dir=MLIR_TMP_DIR, delete=False
    )
    code_file.write(code)
    code_file.close()

    # Build the sbatch command
    cmd: list[str] = ["sbatch", str(MLIR_SCRIPT), code_file.name]

    transform_file = None
    transform_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".mlir", dir=MLIR_TMP_DIR, delete=False
    )
    transform_file.write(BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE)
    transform_file.close()
    cmd.extend(["--transform-file", transform_file.name])

    cmd.extend(["--pass-pipeline", *PASS_PIPELINE])

    # Submit the job
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID: {result.stdout.strip()}")
    job_id = match.group(1)

    # Wait for the job to finish
    timeout = 300
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
        raise TimeoutError(f"SLURM job {job_id} did not finish within {timeout}s")

    # Clean up temp files
    try:
        Path(code_file.name).unlink(missing_ok=True)
        if transform_file:
            Path(transform_file.name).unlink(missing_ok=True)
    except OSError:
        pass

    # Read the output
    log_path = MLIR_SLURM_LOG_DIR / f"{job_id}.out"
    if not log_path.exists():
        raise FileNotFoundError(f"SLURM log not found: {log_path}")
    output = log_path.read_text().strip()
    
    error_path = MLIR_SLURM_LOG_DIR / f"{job_id}.err"
    if error_path.exists():
        error_output = error_path.read_text().strip()

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

@mcp.tool()
def measure_speedup(mlir_base_execution_time: float, mlir_optimized_execution_time: float) -> dict[str, float]:
    """
    Measures the speedup achieved by MLIR transformations.
    
    This tool compares the execution time of base code against transformed code
    to calculate the performance improvement factor.
    
    Use this when you need to:
    - Quantify optimization effectiveness
    - Compare performance before and after transformations
    - Calculate speedup ratios
    
    Args:
        mlir_base_execution_time: Execution time of original (unoptimized) MLIR code in milliseconds
        mlir_optimized_execution_time: Execution time of transformed (optimized) MLIR code in milliseconds
    
    Returns:
        dict with:
            speedup: mlir_base_execution_time / mlir_optimized_execution_time
    """
    result = {
        "speedup": mlir_base_execution_time / mlir_optimized_execution_time
    }
    return result

if __name__ == "__main__":
    mcp.run()
