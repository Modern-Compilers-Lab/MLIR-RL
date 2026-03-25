from llm_action.src.execution.mlir_execution import execute_mlir

class LocalExecutor:
    """In-process MLIR execution (no SLURM, no Dask).

    Runs execute_mlir directly in the main process. BindingsProcess subprocess
    isolation is intentionally disabled: MLIR Python bindings are not fork-safe
    once initialized, so forking after the first transform crashes the child.
    """

    def execute(self, code: str) -> tuple[float, bool]:
        exec_time_ns, success = execute_mlir(code)
        return exec_time_ns / 1_000_000, success
