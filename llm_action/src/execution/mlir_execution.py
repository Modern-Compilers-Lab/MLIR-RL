import argparse
import json
import re
import sys
from typing import Optional

from llm_action.src.utils.transformation import (
    run_transform_code,
    execute_bufferized_code,
    BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE,
    PASS_PIPELINE,
    _execute_from_path_bind_call,
)
from llm_action.src.config import CODE_BUFFERIZE_AND_EXECUTE_TIMEOUT
from utils.bindings_process import BindingsProcess


def _is_bufferized(code: str) -> bool:
    """Check if @main already has memref parameters (already bufferized)."""
    match = re.search(r'func\.func @main\(([^)]+)\)', code)
    if match:
        params = match.group(1)
        return 'memref<' in params and 'tensor<' not in params
    return False


def execute_mlir(
    code: str,
    bufferize_transform_code: Optional[str] = None,
    pass_pipeline: Optional[list[str]] = None
) -> tuple[int, bool]:
    """
    End-to-end: bufferize/lower, then benchmark the MLIR code.

    If the code is already bufferized (memref function signatures),
    the bufferization transform is skipped.

    Returns:
        (execution time in ns, success)
    """
    if pass_pipeline is None:
        pass_pipeline = PASS_PIPELINE

    if _is_bufferized(code):
        return execute_bufferized_code(code, pass_pipeline)

    if bufferize_transform_code is None:
        bufferize_transform_code = BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE

    bufferized = run_transform_code(code, bufferize_transform_code)
    return execute_bufferized_code(
        bufferized,
        pass_pipeline
    )


def execute_mlir_from_path(
    code_path: str,
    bufferize_transform_code: Optional[str] = None,
    pass_pipeline: Optional[list[str]] = None,
) -> tuple[int, bool]:
    """Like `execute_mlir`, but the spawn-child reads the MLIR source from
    `code_path` instead of receiving it through the multiprocessing pipe, and
    runs bufferize+execute in a single BindingsProcess round-trip (the
    bufferized intermediate stays in spawn-child memory). Saves two large
    pickles per call.

    Intended for the Dask worker path, where the code is already on shared
    filesystem (the driver wrote it there to avoid the worker-scheduler hop).
    Other callers should keep using `execute_mlir(code: str, ...)`.

    Returns:
        (execution time in ns, success)
    """
    return BindingsProcess.call(
        _execute_from_path_bind_call,
        code_path,
        bufferize_transform_code,
        pass_pipeline,
        timeout=CODE_BUFFERIZE_AND_EXECUTE_TIMEOUT,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark MLIR code execution (for SLURM compute-node runs)."
    )
    parser.add_argument(
        "code_file",
        type=str,
        help="Path to the MLIR code file to execute.",
    )
    parser.add_argument(
        "--transform-file",
        type=str,
        default=None,
        help="Path to a file containing custom bufferization/lowering transform code. "
             "If omitted, the default bufferization + vector lowering transform is used.",
    )
    parser.add_argument(
        "--pass-pipeline",
        type=str,
        nargs="+",
        default=None,
        help="Custom MLIR pass pipeline as a list of passes. "
             "If omitted, the default lowering pipeline is used.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    code = open(args.code_file).read()

    transform_code_str = None
    if args.transform_file:
        transform_code_str = open(args.transform_file).read()

    try:
        exec_time_ns, success = execute_mlir(
            code,
            bufferize_transform_code=transform_code_str,
            pass_pipeline=args.pass_pipeline,
        )
        print(json.dumps({"execution_time_ms": exec_time_ns / 1_000_000, "success": success}))
    except Exception as e:
        print(json.dumps({"error": str(e), "success": False}), file=sys.stderr)
        sys.exit(1)
