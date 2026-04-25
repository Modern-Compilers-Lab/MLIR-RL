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
    stride=1 and dilation=1 (matching the MLIR template). Returns the median
    execution time in milliseconds.

    Args:
        N: Batch size.
        C: Input channels.
        H, W: Input spatial dimensions.
        F: Output channels (filters).
        KH, KW: Kernel spatial dimensions.
        OH, OW: Output spatial dimensions (used to derive padding).
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

    # Derive symmetric padding from output shape (stride=1, dilation=1)
    pad_h2 = OH - H + KH - 1
    pad_w2 = OW - W + KW - 1
    if pad_h2 < 0 or pad_w2 < 0 or pad_h2 % 2 != 0 or pad_w2 % 2 != 0:
        raise ValueError(
            f"Cannot derive symmetric padding from H={H}, KH={KH}, OH={OH}, "
            f"W={W}, KW={KW}, OW={OW} (stride=1, dilation=1)"
        )
    pad_h, pad_w = pad_h2 // 2, pad_w2 // 2

    inputs = (
        torch.full((N, C, H, W), fill_value, dtype=dtype),
        torch.full((F, C, KH, KW), fill_value, dtype=dtype),
    )

    def _op(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv2d(x, w, stride=1, padding=(pad_h, pad_w), dilation=1)

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

    for sp in (p_matmul, p_conv):
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
    print(result)
