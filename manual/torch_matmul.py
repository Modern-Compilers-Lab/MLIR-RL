import os
import sys
from statistics import median
import torch
import time


def op(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


def main():
    if len(sys.argv) < 2:
        exit(1)
    matmul_mode = int(sys.argv[1])

    torch.set_grad_enabled(False)
    nthreads = int(os.popen('nproc').read().strip())
    torch.set_num_threads(nthreads)
    match matmul_mode:
        case 1:
            inputs = [
                torch.full((24576, 768), 2, dtype=torch.float64),
                torch.full((768, 384), 2, dtype=torch.float64)
            ]
        case 2:
            inputs = [
                torch.full((512, 512), 2, dtype=torch.float64),
                torch.full((512, 512), 2, dtype=torch.float64)
            ]
        case 3:
            inputs = [
                torch.full((256, 512), 2, dtype=torch.float64),
                torch.full((512, 1024), 2, dtype=torch.float64)
            ]
        case _:
            exit(1)

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
