# MCP Tools — Minimal Server

Server: `llm_action.src.mcp.mcp_server_minimal` (`mlir-optimization-tools-minimal`).

This is the minimal-footprint MCP server, intended for the action-generation/training pipelines that don't need ad-hoc transform application or documentation lookup. Compared to the full server ([MCP.md](MCP.md)), it:

- omits `transform_mlir_code` and `delegate_documentation_lookup`,
- exposes a parameter-less `execute_mlir_code` that always uses the project's default bufferization-and-lowering transform plus pass pipeline,
- requires `torch_execution_time` (no default) on `measure_speedup`.

The five PyTorch baseline tools (matmul, conv2d, add, pooling_nchw_max, relu) are identical to the full server and share the same implementation in `llm_action.src.mcp.utils`.

---

## execute_mlir_code(code: str) -> tuple[float, bool]
Submits a SLURM job to execute the given MLIR code on a dedicated compute node and returns the median execution time. Mirrors the torch_*_by_shape tools so that MLIR and PyTorch run on identical hardware with the same resource reservations and thread-affinity settings.

This tool always applies the project default bufferization-and-lowering transform and pass pipeline (the same defaults documented under `execute_mlir_code` in [MCP.md](MCP.md)).

Args:
    code (str): The MLIR code to execute.

Returns:
    tuple[float, bool]: (median execution time in milliseconds, assertion result)

## execute_torch_matmul_by_shape(M: int, K: int, N: int) -> float
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

## execute_torch_conv2d_by_shape(N: int, C: int, H: int, W: int, F: int, KH: int, KW: int, OH: int, OW: int) -> float
Submits a SLURM job to execute a 2D convolution mirroring `linalg.conv_2d_nchw_fchw` (stride=1, dilation=1; padding derived from output shape) using PyTorch JIT on a compute node and returns the median execution time.

Use this to obtain a PyTorch baseline execution time for a given conv2d shape, which can then be compared against MLIR execution times via the measure_speedup tool.

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

## execute_torch_add_by_shape(A: int, B: int, C: int, D: int) -> float
Submits a SLURM job to execute a 4D elementwise add (mirroring `linalg.add`) using PyTorch JIT and returns the median execution time in milliseconds.

Args:
    A, B, C, D: tensor dimensions (both inputs and output have this shape).

Returns:
    float: the median execution time in milliseconds.

## execute_torch_pooling_nchw_max_by_shape(N: int, C: int, H: int, W: int, KH: int, KW: int, OH: int, OW: int) -> float
Submits a SLURM job to execute a 2D max pool mirroring `linalg.pooling_nchw_max` (dilation=1; stride derived from output shape via `OH = (H - KH) // stride + 1`) using PyTorch JIT and returns the median execution time in milliseconds.

Args:
    N, C: batch and channel dimensions.
    H, W: input spatial dimensions.
    KH, KW: kernel spatial dimensions.
    OH, OW: output spatial dimensions (used to derive stride).

Returns:
    float: the median execution time in milliseconds.

## execute_torch_relu_by_shape(shape: list[int]) -> float
Submits a SLURM job to execute an elementwise ReLU (mirroring the linalg.generic ReLU pattern) on a tensor of the given shape using PyTorch JIT, and returns the median execution time in milliseconds.

Args:
    shape: tensor shape, rank-agnostic (e.g. [128, 1024] or [128, 128, 56, 56]).

Returns:
    float: the median execution time in milliseconds.

## measure_speedup(mlir_base_execution_time: float, mlir_optimized_execution_time: float, torch_execution_time: float) -> dict[str, float]
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
    torch_execution_time: execution time of PyTorch baseline in milliseconds (required; if zero, `speedup_to_torch` is reported as `-1`)

Returns:
    dict with:
        speedup: mlir_base_execution_time / mlir_optimized_execution_time
        speedup_to_torch: torch_execution_time / mlir_optimized_execution_time
