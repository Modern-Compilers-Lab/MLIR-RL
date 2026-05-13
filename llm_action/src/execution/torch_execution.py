import argparse
import os
from statistics import median
import torch
import time

def _matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)

def execute_torch_matmul(
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype = torch.float64,
    fill_value: float = 0.0,
    warmup_iters: int = 5,
    bench_iters: int = 5,
) -> float:
    """
    Execute a matrix multiplication (MxK) @ (KxN) using PyTorch JIT and return
    the median execution time.

    Args:
        M: Number of rows of the first matrix.
        K: Shared inner dimension.
        N: Number of columns of the second matrix.
        dtype: Data type for the tensors
        fill_value: Value used to fill the tensors
        warmup_iters: Number of warm-up iterations before benchmarking.
        bench_iters: Number of timed iterations

    Returns:
        float: the median execution time in milliseconds.
    """
    torch.set_grad_enabled(False)
    nthreads = int(os.popen("nproc").read().strip())
    torch.set_num_threads(nthreads)

    inputs = [
        torch.full((M, K), fill_value, dtype=dtype),
        torch.full((K, N), fill_value, dtype=dtype),
    ]

    jit_op = torch.jit.script(_matmul_op, example_inputs=[inputs])

    # Warm-up
    for _ in range(warmup_iters):
        _ = jit_op(*inputs)

    # Benchmark
    times: list[int] = []
    for _ in range(bench_iters):
        start_time = time.perf_counter_ns()
        _ = jit_op(*inputs)
        end_time = time.perf_counter_ns()
        times.append(end_time - start_time)

    median_ns = median(times)
    return median_ns / 1_000_000

def execute_torch_add(
    A: int,
    B: int,
    C: int,
    D: int,
    dtype: torch.dtype = torch.float64,
    fill_value: float = 0.0,
    warmup_iters: int = 5,
    bench_iters: int = 5,
) -> float:
    """
    Execute a 4D elementwise add mirroring `linalg.add` and return the median
    execution time in milliseconds.
    """
    torch.set_grad_enabled(False)
    nthreads = int(os.popen("nproc").read().strip())
    torch.set_num_threads(nthreads)

    inputs = (
        torch.full((A, B, C, D), fill_value, dtype=dtype),
        torch.full((A, B, C, D), fill_value, dtype=dtype),
    )

    def _op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.add(x, y)

    jit_op = torch.jit.trace(_op, example_inputs=inputs)

    for _ in range(warmup_iters):
        _ = jit_op(*inputs)

    times: list[int] = []
    for _ in range(bench_iters):
        start_time = time.perf_counter_ns()
        _ = jit_op(*inputs)
        end_time = time.perf_counter_ns()
        times.append(end_time - start_time)

    return median(times) / 1_000_000

def _derive_stride(in_dim: int, k: int, out_dim: int, axis: str) -> int:
    """Solve for stride such that `out_dim == (in_dim - k) // stride + 1`
    (dilation=1, padding=0). Mirrors the no-padding linalg conv/pool semantics.
    """
    if out_dim <= 0:
        raise ValueError(f"Invalid output {axis} dim: {out_dim}")
    if out_dim == 1:
        stride = max(in_dim - k + 1, 1)
    else:
        stride = (in_dim - k) // (out_dim - 1)
    if stride < 1 or (in_dim - k) // stride + 1 != out_dim:
        raise ValueError(
            f"Cannot derive stride for {axis}: in={in_dim}, k={k}, out={out_dim} "
            f"(dilation=1)"
        )
    return stride

def execute_torch_pooling_nchw_max(
    N: int,
    C: int,
    H: int,
    W: int,
    KH: int,
    KW: int,
    OH: int,
    OW: int,
    dtype: torch.dtype = torch.float64,
    fill_value: float = 0.0,
    warmup_iters: int = 5,
    bench_iters: int = 5,
) -> float:
    """
    Execute a 2D max pool mirroring `linalg.pooling_nchw_max` with dilation=1
    (matching the MLIR template). Stride is derived from the output shape:
    OH = (H - KH) // stride + 1. Returns the median execution time in milliseconds.
    """
    torch.set_grad_enabled(False)
    nthreads = int(os.popen("nproc").read().strip())
    torch.set_num_threads(nthreads)

    stride_h = _derive_stride(H, KH, OH, "H")
    stride_w = _derive_stride(W, KW, OW, "W")

    x = torch.full((N, C, H, W), fill_value, dtype=dtype)

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(
            t, kernel_size=(KH, KW), stride=(stride_h, stride_w), dilation=1
        )

    jit_op = torch.jit.trace(_op, example_inputs=(x,))

    for _ in range(warmup_iters):
        _ = jit_op(x)

    times: list[int] = []
    for _ in range(bench_iters):
        start_time = time.perf_counter_ns()
        _ = jit_op(x)
        end_time = time.perf_counter_ns()
        times.append(end_time - start_time)

    return median(times) / 1_000_000

def execute_torch_relu(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float64,
    fill_value: float = 0.0,
    warmup_iters: int = 5,
    bench_iters: int = 5,
) -> float:
    """
    Execute an elementwise ReLU on a tensor of the given shape (rank-agnostic,
    mirroring the linalg.generic ReLU pattern in the glossary). Returns the
    median execution time in milliseconds.
    """
    torch.set_grad_enabled(False)
    nthreads = int(os.popen("nproc").read().strip())
    torch.set_num_threads(nthreads)

    x = torch.full(tuple(shape), fill_value, dtype=dtype)

    def _op(t: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.relu(t)

    jit_op = torch.jit.trace(_op, example_inputs=(x,))

    for _ in range(warmup_iters):
        _ = jit_op(x)

    times: list[int] = []
    for _ in range(bench_iters):
        start_time = time.perf_counter_ns()
        _ = jit_op(x)
        end_time = time.perf_counter_ns()
        times.append(end_time - start_time)

    return median(times) / 1_000_000

def execute_torch_conv2d(
    N: int,
    C: int,
    H: int,
    W: int,
    F: int,
    KH: int,
    KW: int,
    OH: int,
    OW: int,
    dtype: torch.dtype = torch.float64,
    fill_value: float = 0.0,
    warmup_iters: int = 5,
    bench_iters: int = 5,
) -> float:
    """
    Execute a 2D convolution mirroring `linalg.conv_2d_nchw_fchw` with
    dilation=1 and no padding (linalg conv has no padding attribute). Stride
    is derived per axis from the output shape: OH = (H - KH) // stride + 1.
    Returns the median execution time in milliseconds.

    Args:
        N: Batch size.
        C: Input channels.
        H, W: Input spatial dimensions.
        F: Output channels (filters).
        KH, KW: Kernel spatial dimensions.
        OH, OW: Output spatial dimensions (used to derive stride).
        dtype: Data type for the tensors.
        fill_value: Value used to fill the tensors.
        warmup_iters: Number of warm-up iterations before benchmarking.
        bench_iters: Number of timed iterations.

    Returns:
        float: the median execution time in milliseconds.
    """
    torch.set_grad_enabled(False)
    nthreads = int(os.popen("nproc").read().strip())
    torch.set_num_threads(nthreads)

    stride_h = _derive_stride(H, KH, OH, "H")
    stride_w = _derive_stride(W, KW, OW, "W")

    inputs = (
        torch.full((N, C, H, W), fill_value, dtype=dtype),
        torch.full((F, C, KH, KW), fill_value, dtype=dtype),
    )

    def _op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(
            x, w, stride=(stride_h, stride_w), padding=0, dilation=1
        )

    jit_op = torch.jit.trace(_op, example_inputs=inputs)

    # Warm-up
    for _ in range(warmup_iters):
        _ = jit_op(*inputs)

    # Benchmark
    times: list[int] = []
    for _ in range(bench_iters):
        start_time = time.perf_counter_ns()
        _ = jit_op(*inputs)
        end_time = time.perf_counter_ns()
        times.append(end_time - start_time)

    median_ns = median(times)
    return median_ns / 1_000_000

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch JIT tensor operations."
    )
    subparsers = parser.add_subparsers(dest="op", required=True)

    p_matmul = subparsers.add_parser("matmul", help="2D matrix multiplication (M×K) @ (K×N)")
    p_matmul.add_argument("M", type=int, help="Number of rows of the first matrix")
    p_matmul.add_argument("K", type=int, help="Shared inner dimension")
    p_matmul.add_argument("N", type=int, help="Number of columns of the second matrix")

    p_conv = subparsers.add_parser("conv2d", help="2D convolution (NCHW input, FCHW filter)")
    p_conv.add_argument("N", type=int, help="Batch size")
    p_conv.add_argument("C", type=int, help="Input channels")
    p_conv.add_argument("H", type=int, help="Input height")
    p_conv.add_argument("W", type=int, help="Input width")
    p_conv.add_argument("F", type=int, help="Output channels (filters)")
    p_conv.add_argument("KH", type=int, help="Kernel height")
    p_conv.add_argument("KW", type=int, help="Kernel width")
    p_conv.add_argument("OH", type=int, help="Output height")
    p_conv.add_argument("OW", type=int, help="Output width")

    p_add = subparsers.add_parser("add", help="4D elementwise add")
    p_add.add_argument("A", type=int)
    p_add.add_argument("B", type=int)
    p_add.add_argument("C", type=int)
    p_add.add_argument("D", type=int)

    p_pool = subparsers.add_parser("pooling_nchw_max", help="2D max pool (NCHW)")
    p_pool.add_argument("N", type=int, help="Batch size")
    p_pool.add_argument("C", type=int, help="Channels")
    p_pool.add_argument("H", type=int, help="Input height")
    p_pool.add_argument("W", type=int, help="Input width")
    p_pool.add_argument("KH", type=int, help="Kernel height")
    p_pool.add_argument("KW", type=int, help="Kernel width")
    p_pool.add_argument("OH", type=int, help="Output height")
    p_pool.add_argument("OW", type=int, help="Output width")

    p_relu = subparsers.add_parser("relu", help="Elementwise ReLU on a tensor of arbitrary rank")
    p_relu.add_argument("shape", type=int, nargs="+", help="Tensor shape (e.g. 128 1024 or 128 128 56 56)")

    for sp in (p_matmul, p_conv, p_add, p_pool, p_relu):
        sp.add_argument(
            "--dtype",
            type=str,
            default="float64",
            choices=DTYPE_MAP.keys(),
            help="Data type for the tensors (default: float64)",
        )
        sp.add_argument(
            "--fill-value",
            type=float,
            default=0.0,
            help="Value used to fill the tensors (default: 0.0)",
        )
        sp.add_argument(
            "--warmup-iters",
            type=int,
            default=5,
            help="Number of warm-up iterations (default: 5)",
        )
        sp.add_argument(
            "--bench-iters",
            type=int,
            default=5,
            help="Number of timed iterations (default: 5)",
        )
    return parser.parse_args()

# Examples (run from project root):
#   python -m llm_action.src.execution.torch_execution matmul 256 256 512
#   python -m llm_action.src.execution.torch_execution matmul 256 512 1024 --dtype float32 --bench-iters 10
#   python -m llm_action.src.execution.torch_execution conv2d 128 32 7 7 256 1 1 7 7
#   python -m llm_action.src.execution.torch_execution conv2d 64 64 56 56 128 3 3 56 56 --dtype float32
if __name__ == "__main__":
    args = parse_args()
    common = dict(
        dtype=DTYPE_MAP[args.dtype],
        fill_value=args.fill_value,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    if args.op == "matmul":
        result = execute_torch_matmul(M=args.M, K=args.K, N=args.N, **common)
    elif args.op == "conv2d":
        result = execute_torch_conv2d(
            N=args.N, C=args.C, H=args.H, W=args.W,
            F=args.F, KH=args.KH, KW=args.KW,
            OH=args.OH, OW=args.OW,
            **common,
        )
    elif args.op == "add":
        result = execute_torch_add(A=args.A, B=args.B, C=args.C, D=args.D, **common)
    elif args.op == "pooling_nchw_max":
        result = execute_torch_pooling_nchw_max(
            N=args.N, C=args.C, H=args.H, W=args.W,
            KH=args.KH, KW=args.KW, OH=args.OH, OW=args.OW,
            **common,
        )
    elif args.op == "relu":
        result = execute_torch_relu(shape=tuple(args.shape), **common)
    print(result)
