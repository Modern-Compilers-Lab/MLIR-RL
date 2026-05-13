import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from llm_action.src.config import PROJECT_ROOT, MLIR_SCRIPT, MLIR_TMP_DIR, MLIR_SLURM_LOG_DIR, SLURM_TIMEOUT, SLURM_POLL_INTERVAL

class SlurmExecutor:
    def __init__(self, timeout: int = SLURM_TIMEOUT):
        self.timeout = timeout

    def execute(self, code: str) -> tuple[float, bool]:
        MLIR_TMP_DIR.mkdir(parents=True, exist_ok=True)
        MLIR_SLURM_LOG_DIR.mkdir(parents=True, exist_ok=True)

        code_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", dir=MLIR_TMP_DIR, delete=False
        )
        code_file.write(code)
        code_file.close()

        try:
            result = subprocess.run(
                ["sbatch", str(MLIR_SCRIPT), code_file.name],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            if result.returncode != 0:
                raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if not match:
                raise RuntimeError(f"Could not parse job ID: {result.stdout.strip()}")
            job_id = match.group(1)

            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                sq = subprocess.run(
                    ["squeue", "-j", job_id, "-h", "-o", "%T"],
                    capture_output=True, text=True,
                )
                if not sq.stdout.strip():
                    break
                time.sleep(SLURM_POLL_INTERVAL)
            else:
                raise TimeoutError(f"SLURM job {job_id} timed out after {self.timeout}s")

            log_path = MLIR_SLURM_LOG_DIR / f"{job_id}.out"
            if not log_path.exists():
                raise FileNotFoundError(f"Log not found: {log_path}")

            output = log_path.read_text().strip()
            err_path = MLIR_SLURM_LOG_DIR / f"{job_id}.err"
            err = err_path.read_text().strip() if err_path.exists() else ""

            try:
                data = json.loads(output.splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                raise RuntimeError(f"Bad output from job {job_id}:\n{output}\n{err}")

            if "error" in data:
                raise RuntimeError(f"Job {job_id} failed:\n{output}\n{err}")

            return data["execution_time_ms"], data["success"]

        finally:
            Path(code_file.name).unlink(missing_ok=True)

    def execute_torch(self, op_type: str, dims: tuple[int, ...]) -> float:
        from llm_action.src.mcp.utils import run_torch_sbatch
        return run_torch_sbatch([op_type, *map(str, dims)])
