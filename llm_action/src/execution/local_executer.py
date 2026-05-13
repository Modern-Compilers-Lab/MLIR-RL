from llm_action.src.execution.mlir_execution import execute_mlir

class LocalExecutor:
    """In-process MLIR execution (no SLURM, no Dask).

    Runs execute_mlir directly in the main process. BindingsProcess uses spawn
    context (not fork) to isolate MLIR C++ bindings in a child process, avoiding
    fork-safety issues while protecting the parent from segfaults.
    """

    def execute(self, code: str) -> tuple[float, bool]:
        exec_time_ns, success = execute_mlir(code)
        return exec_time_ns / 1_000_000, success

    def execute_torch(self, op_type: str, dims: tuple[int, ...]) -> float:
        from llm_action.src.execution.torch_execution import (
            execute_torch_matmul, execute_torch_conv2d, execute_torch_add,
            execute_torch_pooling_nchw_max, execute_torch_relu,
        )
        if op_type == "matmul":
            return execute_torch_matmul(*dims)
        if op_type == "conv2d":
            return execute_torch_conv2d(*dims)
        if op_type == "add":
            return execute_torch_add(*dims)
        if op_type == "pooling_nchw_max":
            return execute_torch_pooling_nchw_max(*dims)
        if op_type == "relu":
            return execute_torch_relu(tuple(dims))
        raise ValueError(f"Unknown torch op_type: {op_type}")
