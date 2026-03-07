import argparse
import json
import os
from pathlib import Path
from statistics import median
import torch
import time

PARENT_DIR = Path(__file__).parents[1]


def matmul_op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


def matmul_inputs(size: dict[str, int]) -> list[torch.Tensor]:
    return [
        torch.full((size['M'], size['K']), 2, dtype=torch.float64),
        torch.full((size['K'], size['N']), 2, dtype=torch.float64)
    ]


def conv_2d_op(input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv2d(input, weight)


def conv_2d_inputs(size: dict[str, int]) -> list[torch.Tensor]:
    return [
        torch.full((size['N'], size['C'], size['H'], size['W']), 2, dtype=torch.float64),
        torch.full((size['F'], size['C'], size['KH'], size['KW']), 2, dtype=torch.float64)
    ]


def main():
    parser = argparse.ArgumentParser(description='Run PyTorch matmul with specified id.')
    parser.add_argument('id', type=str, help='The unique identifier for the MLIR code to transform. It takes the form "{name}_{instance}", where "name" is the name of the benchmark (e.g. "matmul") and "instance" is the specific instance (e.g. "0", "1", etc.).')
    args = parser.parse_args()
    name, instance = args.id.rsplit("_", 1)

    with open(PARENT_DIR / "data" / name / "sizes.json", 'r') as f:
        sizes = json.load(f)
    size = sizes[instance]

    match name:
        case "matmul":
            op = matmul_op
            inputs = matmul_inputs(size)
        case "conv_2d":
            op = conv_2d_op
            inputs = conv_2d_inputs(size)
        case _:
            raise ValueError(f"Unsupported benchmark name: {name}")

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
