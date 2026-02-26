import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from fastmcp import FastMCP
from llm_action.src.utils.transformation import run_transform_code, BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE, PASS_PIPELINE
from llm_action.src.agents.documentation_lookup import DocumentationLookupAgentWrapper
from llm_action.src.models import KernelType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TORCH_SCRIPT = PROJECT_ROOT / "llm_action" / "scripts" / "torch.sh"
MLIR_SCRIPT  = PROJECT_ROOT / "llm_action" / "scripts" / "mlir.sh"
SLURM_LOG_DIR = PROJECT_ROOT / "llm_action" / "logs" / "jobs"
MLIR_TMP_DIR  = PROJECT_ROOT / "llm_action" / "tmp"

mcp = FastMCP("mlir-optimization-tools")

@mcp.tool()
def transform_mlir_code(code: str, transformation_code: str) -> str:
    """
    Applies MLIR transformations to the given code using custom transformation scripts.
    
    This tool takes base MLIR code and applies user-defined transformations to it,
    allowing for optimization passes, dialect conversions, or other code modifications.
    
    Use this when you need to:
    - Apply specific MLIR transformation passes to code
    - Test different optimization strategies
    - Convert between MLIR dialects
    - Modify MLIR operations programmatically
    
    Args:
        code: The base MLIR code to transform
        transformation_code: The transformation script/pass to apply
    
    Returns:
        The transformed MLIR code as a string
    """
    return run_transform_code(code, transformation_code)

@mcp.tool()
def execute_mlir_code(code: str, bufferization_lowering_v_transform_code: str | None = None, pass_pipeline: list[str] | None = None) -> tuple[float, bool]:
    F"""
    Submits a SLURM job to execute the given MLIR code on a dedicated compute
    node and returns the median execution time.  This mirrors
    execute_torch_matmul_by_shape to ensure fair benchmarking: both MLIR and
    PyTorch run on identical hardware with the same resource reservations and
    thread-affinity settings.

    Args:
        code (str): The MLIR code to execute.
        bufferization_lowering_v_transform_code (Optional[str]): Optional
            transformation code to apply for bufferization and lowering before execution.
        pass_pipeline (Optional[list[str]]): Optional list of MLIR passes to apply during execution. Do not include a wrapper, that's handled internally, basically provide the list of passes you want to run in the form of ["pass1", "pass2", ...].

    Defaults (Current Functional Behavior):
        bufferization_lowering_v_transform_code:
        ```
        {BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE}
        ```

        pass_pipeline:
        ```
        {PASS_PIPELINE}
        ```

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
    if bufferization_lowering_v_transform_code is not None:
        transform_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", dir=MLIR_TMP_DIR, delete=False
        )
        transform_file.write(bufferization_lowering_v_transform_code)
        transform_file.close()
        cmd.extend(["--transform-file", transform_file.name])

    if pass_pipeline is not None:
        cmd.extend(["--pass-pipeline", *pass_pipeline])

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
    log_path = SLURM_LOG_DIR / f"mlir_{job_id}.out"
    if not log_path.exists():
        raise FileNotFoundError(f"SLURM log not found: {log_path}")
    output = log_path.read_text().strip()
    
    error_path = SLURM_LOG_DIR / f"mlir_{job_id}.err"
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
def execute_torch_matmul_by_shape(M: int, K: int, N: int) -> float:
    """
    Submits a SLURM job to execute a matrix multiplication (M×K) @ (K×N)
    using PyTorch JIT on a compute node and returns the median execution time.

    Use this to obtain a PyTorch baseline execution time for a given matrix
    multiplication shape, which can then be compared against MLIR execution times
    via the measure_speedup tool.

    Args:
        M: Number of rows of the first matrix.
        K: Shared inner dimension (columns of first matrix / rows of second matrix).
        N: Number of columns of the second matrix.

    Returns:
        float: the median execution time in milliseconds.
    """
    # Submit the job
    result = subprocess.run(
        ["sbatch", str(TORCH_SCRIPT), str(M), str(K), str(N)],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
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

    # Read the output
    log_path = SLURM_LOG_DIR / f"torch_{job_id}.out"
    if not log_path.exists():
        raise FileNotFoundError(f"SLURM log not found: {log_path}")
    output = log_path.read_text().strip()

    try:
        return float(output.splitlines()[-1])
    except (ValueError, IndexError):
        raise RuntimeError(f"Could not parse execution time from job {job_id} output:\n{output}")

@mcp.tool()
def measure_speedup(mlir_base_execution_time: float, mlir_optimized_execution_time: float, torch_execution_time: float) -> dict[str, float]:
    """
    Measures the speedup achieved by MLIR transformations.
    
    This tool compares the execution time of base code against transformed code
    to calculate the performance improvement factor. It compares against
    a PyTorch baseline to compute the speedup relative to PyTorch.
    
    Use this when you need to:
    - Quantify optimization effectiveness
    - Compare performance before and after transformations
    - Calculate speedup ratios
    - Evaluate transformation impact relative to PyTorch
    
    Args:
        mlir_base_execution_time: Execution time of original (unoptimized) MLIR code in milliseconds
        mlir_optimized_execution_time: Execution time of transformed (optimized) MLIR code in milliseconds
        torch_execution_time: execution time of PyTorch baseline in milliseconds
    
    Returns:
        dict with:
            speedup: mlir_base_execution_time / mlir_optimized_execution_time
            speedup_to_torch: torch_execution_time / mlir_optimized_execution_time  
    """
    result = {
        "speedup": mlir_base_execution_time / mlir_optimized_execution_time,
        "speedup_to_torch": torch_execution_time / mlir_optimized_execution_time
    }
    return result

@mcp.tool()
def delegate_documentation_lookup(task: str) -> str:
    """
    Delegates a documentation lookup task to the Documentation Lookup Agent.
    
    This tool forwards a specific documentation retrieval task to the Documentation Lookup Agent, which specializes in finding authoritative references for MLIR Transform dialect operations.
    
    Args:
        task: The documentation lookup task or question to be answered
        
    Returns:
        The response from the Documentation Lookup Agent containing the requested documentation information
    """
    agent = DocumentationLookupAgentWrapper()
    result = agent.run(task)
    return result

if __name__ == "__main__":
    mcp.run()

# @mcp.tool()
# def execute_mlir_code(code: str, bufferization_lowering_v_transform_code: str | None = None, pass_pipeline: list[str] | None = None) -> tuple[float, bool]:
#     F"""
#     Executes a given MLIR code with a timeout.

#     Args:
#         code (str): The MLIR code to execute
#         bufferization_lowering_v_transform_code (Optional[str]): Optional transformation code to apply for bufferization and lowering before execution.
#         pass_pipeline (Optional[list[str]]): Optional list of MLIR passes to apply during execution.
            
#     Defaults (Current Functional Behavior):
#         bufferization_lowering_v_transform_code:
#         ```
#         {BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE}
#         ```
        
#         pass_pipeline:
#         ```
#         {PASS_PIPELINE}
#         ```

#     Returns:
#         tuple[float, bool]: (execution time in milliseconds, assertion result)
#     """
#     bufferized = transform_bufferize_and_lower_v(code, transform_code=bufferization_lowering_v_transform_code)
#     exec_time_ns, success = execute_bufferized_code(bufferized, pass_pipeline=pass_pipeline)
#     return exec_time_ns / 1_000_000, success


# @mcp.tool()
# def list_transformations() -> str:
#     """Lists all available MLIR Transform dialect transformation categories and names."""
#     doc = load_documentation()
#     lines = []
#     for category, transforms in doc.items():
#         lines.append(f"## {category}")
#         for name in transforms:
#             lines.append(f"  - {name}")
#     return "\n".join(lines)

# @mcp.tool()
# def lookup_transformation(category_name: str, transformation_name: str) -> str:
#     """
#     Looks up MLIR Transform dialect documentation for a specific transformation.
    
#     This tool retrieves detailed documentation for a given transformation from the MLIR Transform dialect reference, including operation names, required operands/results, attributes, and example snippets.
    
#     Args:
#         category_name: The name of the transformation category
#         transformation_name: The name of the specific transformation
    
#     Returns:
#         Detailed documentation string for the specified transformation
#     """
#     doc = load_documentation()
#     return doc[category_name][transformation_name]