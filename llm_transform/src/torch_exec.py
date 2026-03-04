import argparse
import json
import os
from pathlib import Path
from statistics import median
import torch
import time

PARENT_DIR = Path(__file__).parents[1]


def op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


def main():
    parser = argparse.ArgumentParser(description='Run PyTorch matmul with specified id.')
    parser.add_argument('--id', type=int, required=True, help='The unique identifier for the MLIR code to run.')
    args = parser.parse_known_args()[0]

    with open(PARENT_DIR / "data" / "matmul" / "sizes.json", 'r') as f:
        sizes = json.load(f)
    matmul_size = sizes[str(args.id)]
    inputs = [
        torch.full((matmul_size['M'], matmul_size['K']), 2, dtype=torch.float64),
        torch.full((matmul_size['K'], matmul_size['N']), 2, dtype=torch.float64)
    ]

    torch.set_grad_enabled(False)
    nthreads = int(os.popen('nproc').read().strip())
    torch.set_num_threads(nthreads)

    jit_op = torch.jit.script(op, example_inputs=[inputs])
    for _ in range(10):
        _ = jit_op(*inputs)

    times = []
    for _ in range(11):
        start_time = time.perf_counter_ns()
        _ = jit_op(*inputs)
        end_time = time.perf_counter_ns()
        times.append(end_time - start_time)
    print(median(times))


if __name__ == "__main__":
    main()
