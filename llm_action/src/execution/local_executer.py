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
