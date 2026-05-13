from typing import Optional

from fastmcp import FastMCP

from llm_action.src.utils.transformation import (
    run_transform_code,
    BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE,
    PASS_PIPELINE,
)
from llm_action.src.agents.documentation_lookup import DocumentationLookupAgentWrapper
from llm_action.src.mcp.utils import (
    run_mlir_sbatch,
    torch_matmul_by_shape,
    torch_conv2d_by_shape,
    torch_add_by_shape,
    torch_pooling_nchw_max_by_shape,
    torch_relu_by_shape,
    compute_speedup,
)

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
def execute_mlir_code(
    code: str,
    bufferization_lowering_v_transform_code: str | None = None,
    pass_pipeline: list[str] | None = None,
) -> tuple[float, bool]:
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
    return run_mlir_sbatch(
        code,
        bufferization_lowering_v_transform_code=bufferization_lowering_v_transform_code,
        pass_pipeline=pass_pipeline,
    )

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
    return torch_matmul_by_shape(M, K, N)

@mcp.tool()
def execute_torch_conv2d_by_shape(
    N: int, C: int, H: int, W: int,
    F: int, KH: int, KW: int, OH: int, OW: int,
) -> float:
    """
    Submits a SLURM job to execute a 2D convolution mirroring
    `linalg.conv_2d_nchw_fchw` (dilation=1, no padding; stride derived per
    axis from output shape via `OH = (H - KH) // stride + 1`) using PyTorch
    JIT and returns the median execution time.

    Use this to obtain a PyTorch baseline execution time for a given conv2d
    shape, which can then be compared against MLIR execution times via the
    measure_speedup tool.

    Args:
        N: Batch size.
        C: Input channels.
        H: Input height.
        W: Input width.
        F: Output channels (filters).
        KH: Kernel height.
        KW: Kernel width.
        OH: Output height.
        OW: Output width.

    Returns:
        float: the median execution time in milliseconds.
    """
    return torch_conv2d_by_shape(N, C, H, W, F, KH, KW, OH, OW)

@mcp.tool()
def execute_torch_add_by_shape(A: int, B: int, C: int, D: int) -> float:
    """
    Submits a SLURM job to execute a 4D elementwise add (mirroring `linalg.add`)
    using PyTorch JIT and returns the median execution time in milliseconds.

    Args:
        A, B, C, D: tensor dimensions (both inputs and output have this shape).

    Returns:
        float: the median execution time in milliseconds.
    """
    return torch_add_by_shape(A, B, C, D)

@mcp.tool()
def execute_torch_pooling_nchw_max_by_shape(
    N: int, C: int, H: int, W: int,
    KH: int, KW: int, OH: int, OW: int,
) -> float:
    """
    Submits a SLURM job to execute a 2D max pool mirroring
    `linalg.pooling_nchw_max` (dilation=1; stride derived from output shape via
    `OH = (H - KH) // stride + 1`) using PyTorch JIT and returns the median
    execution time in milliseconds.

    Args:
        N, C: batch and channel dimensions.
        H, W: input spatial dimensions.
        KH, KW: kernel spatial dimensions.
        OH, OW: output spatial dimensions (used to derive stride).

    Returns:
        float: the median execution time in milliseconds.
    """
    return torch_pooling_nchw_max_by_shape(N, C, H, W, KH, KW, OH, OW)

@mcp.tool()
def execute_torch_relu_by_shape(shape: list[int]) -> float:
    """
    Submits a SLURM job to execute an elementwise ReLU (mirroring the
    linalg.generic ReLU pattern) on a tensor of the given shape using PyTorch
    JIT, and returns the median execution time in milliseconds.

    Args:
        shape: tensor shape, rank-agnostic (e.g. [128, 1024] or [128, 128, 56, 56]).

    Returns:
        float: the median execution time in milliseconds.
    """
    return torch_relu_by_shape(shape)

@mcp.tool()
def measure_speedup(
    mlir_base_execution_time: float,
    mlir_optimized_execution_time: float,
    torch_execution_time: Optional[float] = None,
) -> dict[str, float]:
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
        torch_execution_time (Optional): execution time of PyTorch baseline in milliseconds

    Returns:
        dict with:
            speedup: mlir_base_execution_time / mlir_optimized_execution_time
            speedup_to_torch: torch_execution_time / mlir_optimized_execution_time
    """
    return compute_speedup(
        mlir_base_execution_time, mlir_optimized_execution_time, torch_execution_time
    )

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
