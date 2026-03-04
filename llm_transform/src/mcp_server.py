from pathlib import Path
import re
import subprocess
import tempfile
from time import time

from fastmcp import FastMCP
from llm_transform.src.utils.transformation import transform_and_lower

PARENT_DIR = Path(__file__).parents[1]

mcp = FastMCP("mlir-transform")


@mcp.tool()
def run_schedule(
    id: int, transform_schedule: str, mlir_passes: str,
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
        id (int): The unique identifier for the MLIR code to transform.
        transform_schedule (str): The transformation schedule to apply, specified as a string of MLIR transformations.
        mlir_passes (str, optional): The MLIR passes to apply during lowering. Defaults to BASE_MLIR_PASSES.
        llvm_passes (str, optional): LLVM opt pass pipeline to run before JIT (e.g. "licm,loop-unroll"). Defaults to "default<O3>".
        llvm_flags (str, optional): Comma-separated LLVM CL flags (e.g. "enable-loop-versioning-licm,licm-mssa-max-acc-promotion=1000"). Defaults to empty string.
        llc_flags (str, optional): Comma-separated flags for llc codegen (e.g. "align-loops=32,enable-split-loopiv-heuristic"). Defaults to empty string.
        bufferize_first (bool, optional): Whether to apply bufferization before applying the transformation schedule. Defaults to True.

    Returns:
        dict: the measured slowdown compared to PyTorch optimized code.
    """

    transform_schedule_tmp = tempfile.NamedTemporaryFile(suffix=".mlir", mode="w")
    transform_schedule_tmp.write(transform_schedule)

    mlir_passes_tmp = tempfile.NamedTemporaryFile(suffix=".txt", mode="w")
    mlir_passes_tmp.write(mlir_passes)

    exec_script = PARENT_DIR / "scripts" / "execute.sh"

    cmd = [
        'sbatch', '--parsable', str(exec_script),
        '--id', str(id),
        '--transform_schedule_file', transform_schedule_tmp.name,
        '--mlir_passes_file', mlir_passes_tmp.name,
        '--llvm_passes', llvm_passes,
        '--llvm_flags', llvm_flags,
        '--llc_flags', llc_flags
    ]

    if not bufferize_first:
        cmd.append('--no-bufferize')

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip()

    # Wait for the job to complete and get the output
    while True:
        sacct_result = subprocess.run(['squeue', '-j', job_id, '-h'], capture_output=True, text=True)
        if sacct_result.returncode != 0:
            raise RuntimeError(f"squeue command failed: {sacct_result.stderr}")
        if sacct_result.stdout.strip() == "":
            break
        time.sleep(1)

    # Read the results of the job
    output_file = PARENT_DIR / "logs" / f"{job_id}.out"
    error_file = PARENT_DIR / "logs" / f"{job_id}.err"

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

    return float(match.group(1))


@mcp.tool()
def lower_schedule(
    id: int, transform_schedule: str, mlir_passes: str,
    llvm_passes: str = "default<O3>", llvm_flags: str = "", llc_flags: str = "",
    bufferize_first: bool = True
) -> dict[str, str]:
    """
    Apply the given transformation schedule to the MLIR code associated with the given ID. Lower using
    the specified MLIR and LLVM passes. Compile to assembly. Return the transformed MLIR ("mlir_transformed"),
    the resulting LLVM ("llvm"), the optimized LLVM ("llvm_opt"), and the generated assembly ("asm").

    Args:
        id (int): The unique identifier for the MLIR code to transform.
        transform_schedule (str): The transformation schedule to apply, specified as a string of MLIR transformations.
        mlir_passes (str, optional): The MLIR passes to apply during lowering. Defaults to BASE_MLIR_PASSES.
        llvm_passes (str, optional): LLVM opt pass pipeline to run before JIT (e.g. "licm,loop-unroll"). Defaults to "default<O3>".
        llvm_flags (str, optional): Comma-separated LLVM CL flags (e.g. "enable-loop-versioning-licm,licm-mssa-max-acc-promotion=1000"). Defaults to empty string.
        llc_flags (str, optional): Comma-separated flags for llc codegen (e.g. "align-loops=32,enable-split-loopiv-heuristic"). Defaults to empty string.
        bufferize_first (bool, optional): Whether to apply bufferization before applying the transformation schedule. Defaults to True.

    Returns:
        dict: a dictionary containing:
            - "mlir_transformed": the transformed MLIR code after applying the transformation schedule
            - "llvm": the generated LLVM IR after lowering the transformed MLIR
            - "llvm_opt": the generated LLVM IR after applying llvm opt with the specified passes and flags
            - "asm": the generated assembly code after compiling with llc and the specified flags
    """

    return transform_and_lower(id, transform_schedule, mlir_passes, llvm_passes, llvm_flags, llc_flags, bufferize_first)
