import json
from datetime import datetime
from pathlib import Path
import re
import subprocess
import tempfile
import time

from fastmcp import FastMCP

from utils.transformation import transform_and_lower

PARENT_DIR = Path(__file__).parents[1]
_LOG_FILE = PARENT_DIR / "logs" / "claude_optimization.log"
_BEST_DIR = PARENT_DIR / "logs" / "best"
_BEST_STATE_FILE = _BEST_DIR / "state.json"

mcp = FastMCP("mlir-transform")


def _load_best_state() -> dict[str, float]:
    if _BEST_STATE_FILE.exists():
        with open(_BEST_STATE_FILE, 'r') as f:
            return json.load(f)
    return {}


def _save_best_state(best_state: dict[str, float]):
    _BEST_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(_BEST_STATE_FILE, 'w') as f:
        json.dump(best_state, f, indent=2)


def _save_best_config(id: str, transform_schedule: str, mlir_passes: str,
                      llvm_passes: str, llvm_flags: str, llc_flags: str):
    name, instance = id.rsplit("_", 1)
    config_dir = _BEST_DIR / name / instance
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "schedule.mlir").write_text(transform_schedule)
    (config_dir / "passes.txt").write_text(mlir_passes)
    lines = [f"llvm_passes={llvm_passes}"]
    if llvm_flags:
        lines.append(f"llvm_flags={llvm_flags}")
    if llc_flags:
        lines.append(f"llc_flags={llc_flags}")
    (config_dir / "llvm-llc-passes-flags.txt").write_text("\n".join(lines) + "\n")


@mcp.tool()
def run_schedule(
    id: str, transform_schedule: str, mlir_passes: str,
    llvm_passes: str = "default<O3>", llvm_flags: str = "", llc_flags: str = "",
    bufferize_first: bool = True
) -> float:
    """
    Apply the given transformation schedule to the MLIR code associated with the given ID. Lower using
    the specified MLIR and LLVM passes. Execute the resulting LLVM code and return the measured
    slowdown compared to PyTorch optimized code.

    This is the main metric for evaluating the effectiveness of the transformation schedule. The slowdown
    must be minimized as much as possible, ideally reaching around 0.5x or better (i.e. the transformed code
    runs at least 2x faster than PyTorch).

    Args:
        id (str): The unique identifier for the MLIR code to transform. It takes the form "{name}_{instance}", where "name" is the name of the benchmark (e.g. "matmul") and "instance" is the specific instance (e.g. "0", "1", etc.).
        transform_schedule (str): The transformation schedule to apply, specified as a string of MLIR transformations.
        mlir_passes (str, optional): The MLIR passes to apply during lowering. Defaults to BASE_MLIR_PASSES.
        llvm_passes (str, optional): LLVM opt pass pipeline to run before JIT (e.g. "licm,loop-unroll"). Defaults to "default<O3>".
        llvm_flags (str, optional): Comma-separated LLVM CL flags (e.g. "enable-loop-versioning-licm,licm-mssa-max-acc-promotion=1000"). Defaults to empty string.
        llc_flags (str, optional): Comma-separated flags for llc codegen (e.g. "align-loops=32,enable-split-loopiv-heuristic"). Defaults to empty string.
        bufferize_first (bool, optional): Whether to apply bufferization before applying the transformation schedule. Defaults to True.

    Returns:
        dict: the measured slowdown compared to PyTorch optimized code.
    """

    tmp_dir = PARENT_DIR / "tmp"
    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="w", delete=False, dir=tmp_dir) as transform_schedule_tmp, \
         tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, dir=tmp_dir) as mlir_passes_tmp:
        transform_schedule_tmp.write(transform_schedule)
        transform_schedule_tmp.flush()

        mlir_passes_tmp.write(mlir_passes)
        mlir_passes_tmp.flush()

    exec_script_name = "execute"
    exec_script = PARENT_DIR / "scripts" / f"{exec_script_name}.sh"

    cmd = [
        'sbatch', '--parsable', str(exec_script),
        '--id', id,
        '--transform_schedule_file', transform_schedule_tmp.name,
        '--mlir_passes_file', mlir_passes_tmp.name,
        '--llvm_passes', llvm_passes,
    ]

    if llvm_flags:
        cmd.extend(['--llvm_flags', llvm_flags])
    if llc_flags:
        cmd.extend(['--llc_flags', llc_flags])
    if not bufferize_first:
        cmd.append('--no-bufferize')

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip()

    # Wait for the job to complete and get the output
    while True:
        squeue_result = subprocess.run(['squeue', '-j', job_id, '-h'], capture_output=True, text=True)
        if squeue_result.returncode != 0:
            raise RuntimeError(f"squeue command failed: {squeue_result.stderr}")
        if squeue_result.stdout.strip() == "":
            break
        time.sleep(1)

    # Delete the temporary files
    Path(transform_schedule_tmp.name).unlink()
    Path(mlir_passes_tmp.name).unlink()

    # Read the results of the job
    output_file = PARENT_DIR / "logs" / "jobs" / f"{exec_script_name}_{job_id}.out"
    error_file = PARENT_DIR / "logs" / "jobs" / f"{exec_script_name}_{job_id}.err"

    # Check for errors first
    if error_file.exists() and error_file.stat().st_size > 0:
        with open(error_file, 'r') as f:
            error_content = f.read()
        raise RuntimeError(f"Job {job_id} failed with error:\n{error_content}")

    # Read the output and extract the slowdown compared to PyTorch
    if not output_file.exists():
        raise RuntimeError(f"Output file {output_file} not found for job {job_id}")
    with open(output_file, 'r') as f:
        output_content = f.read().strip()
    last_line = output_content.splitlines()[-1]
    match = re.search(r"Slowdown compared to PyTorch: ([\d.]+)x", last_line)
    if not match:
        raise RuntimeError(f"Unexpected output format in job {job_id} output: {output_content}")
    slowdown = float(match.group(1))

    # Clean up the output files
    output_file.unlink(missing_ok=True)
    error_file.unlink(missing_ok=True)

    return slowdown


@mcp.tool()
def log_iteration(
    id: str,
    iteration: int,
    slowdown: float = -1.0,
    summary: str = "",
    transform_schedule: str = "",
    mlir_passes: str = "",
    llvm_passes: str = "default<O3>",
    llvm_flags: str = "",
    llc_flags: str = "",
    error: str = "",
) -> str:
    """
    Log an optimization iteration result. Call this after every run_schedule attempt.

    Appends the result to logs/claude_optimization.log in real time.
    If the slowdown is a new best for the benchmark, automatically saves the full
    configuration to logs/best/<name>/<instance>/.

    Args:
        id: Benchmark ID (e.g. "matmul_1").
        iteration: Iteration number for this benchmark.
        slowdown: Measured slowdown vs PyTorch. Use -1 (default) if the run failed.
        summary: Brief description of what was tried in this iteration.
        transform_schedule: The transform schedule used.
        mlir_passes: The MLIR passes used.
        llvm_passes: The LLVM passes used.
        llvm_flags: LLVM flags used.
        llc_flags: LLC flags used.
        error: Error message if the run failed.

    Returns:
        Status message indicating whether this was a new best result.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    if error or slowdown < 0:
        entry = f"\n[{timestamp}] {id} | iter={iteration} | ERROR | {summary}\n"
        if error:
            entry += f"  Error: {error}\n"
        with open(_LOG_FILE, 'a') as f:
            f.write(entry)
        return "Logged (error, no slowdown recorded)"

    slowdown_str = f"{slowdown:.4f}x"
    entry = f"\n[{timestamp}] {id} | iter={iteration} | {slowdown_str} | {summary}\n"

    best_state = _load_best_state()
    is_new_best = id not in best_state or slowdown < best_state[id]
    if is_new_best:
        old_best = best_state.get(id)
        best_state[id] = slowdown
        _save_best_state(best_state)
        _save_best_config(id, transform_schedule, mlir_passes, llvm_passes, llvm_flags, llc_flags)
        if old_best is not None:
            entry += f"  *** NEW BEST for {id} (previous: {old_best:.4f}x) ***\n"
        else:
            entry += f"  *** FIRST RESULT for {id} ***\n"

    with open(_LOG_FILE, 'a') as f:
        f.write(entry)

    if is_new_best:
        if old_best is not None:
            msg = f"New best for {id}: {slowdown_str} (previous: {old_best:.4f}x)"
        else:
            msg = f"First result for {id}: {slowdown_str}"
        msg += " — config saved to logs/best/"
        return msg

    return f"Logged. Current best for {id}: {best_state[id]:.4f}x"


@mcp.tool()
def lower_schedule(
    id: str, transform_schedule: str, mlir_passes: str,
    llvm_passes: str = "default<O3>", llvm_flags: str = "", llc_flags: str = "",
    bufferize_first: bool = True
) -> dict[str, str]:
    """
    Apply the given transformation schedule to the MLIR code associated with the given ID. Lower using
    the specified MLIR and LLVM passes. Compile to assembly. Save intermediate outputs to out/<id>/ and
    return file paths for the transformed MLIR, LLVM IR, optimized LLVM IR, and assembly.

    Args:
        id (str): The unique identifier for the MLIR code to transform. It takes the form "{name}_{instance}", where "name" is the name of the benchmark (e.g. "matmul") and "instance" is the specific instance (e.g. "0", "1", etc.).
        transform_schedule (str): The transformation schedule to apply, specified as a string of MLIR transformations.
        mlir_passes (str, optional): The MLIR passes to apply during lowering. Defaults to BASE_MLIR_PASSES.
        llvm_passes (str, optional): LLVM opt pass pipeline to run before JIT (e.g. "licm,loop-unroll"). Defaults to "default<O3>".
        llvm_flags (str, optional): Comma-separated LLVM CL flags (e.g. "enable-loop-versioning-licm,licm-mssa-max-acc-promotion=1000"). Defaults to empty string.
        llc_flags (str, optional): Comma-separated flags for llc codegen (e.g. "align-loops=32,enable-split-loopiv-heuristic"). Defaults to empty string.
        bufferize_first (bool, optional): Whether to apply bufferization before applying the transformation schedule. Defaults to True.

    Returns:
        dict: a dictionary containing file paths to the generated outputs:
            - "mlir_transformed": path to the transformed MLIR file after applying the transformation schedule
            - "llvm": path to the generated LLVM IR file after lowering the transformed MLIR
            - "llvm_opt": path to the generated LLVM IR file after applying llvm opt with the specified passes and flags
            - "asm": path to the generated assembly file after compiling with llc and the specified flags
    """

    return transform_and_lower(id, transform_schedule, mlir_passes, llvm_passes, llvm_flags, llc_flags, bufferize_first)


if __name__ == "__main__":
    mcp.run()
