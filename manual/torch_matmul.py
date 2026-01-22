from statistics import median
import torch
import time


def op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)


def main():
    inputs = [
        # torch.rand(24576, 768, dtype=torch.float64),
        # torch.rand(768, 384, dtype=torch.float64)
        # Array filled with 2
        torch.full((24576, 768), 2, dtype=torch.float64),
        torch.full((768, 384), 2, dtype=torch.float64)
    ]
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
