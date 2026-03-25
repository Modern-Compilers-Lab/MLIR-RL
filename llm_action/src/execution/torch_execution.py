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

DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch JIT matrix multiplication."
    )
    parser.add_argument("M", type=int, help="Number of rows of the first matrix")
    parser.add_argument("K", type=int, help="Shared inner dimension")
    parser.add_argument("N", type=int, help="Number of columns of the second matrix")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float64",
        choices=DTYPE_MAP.keys(),
        help="Data type for the tensors (default: float64)",
    )
    parser.add_argument(
        "--fill-value",
        type=float,
        default=0.0,
        help="Value used to fill the tensors (default: 0.0)",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=5,
        help="Number of warm-up iterations (default: 5)",
    )
    parser.add_argument(
        "--bench-iters",
        type=int,
        default=5,
        help="Number of timed iterations (default: 5)",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    result = execute_torch_matmul(
        M=args.M,
        K=args.K,
        N=args.N,
        dtype=DTYPE_MAP[args.dtype],
        fill_value=args.fill_value,
        warmup_iters=args.warmup_iters,
        bench_iters=args.bench_iters,
    )
    print(result)
