import argparse
import json
import sys
from typing import Optional

from llm_action.src.utils.transformation import run_transform_code, execute_bufferized_code, BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE, PASS_PIPELINE

def execute_mlir(
    code: str,
    bufferize_transform_code: Optional[str] = None,
    pass_pipeline: Optional[list[str]] = None
) -> tuple[int, bool]:
    """
    End-to-end: bufferize/lower, then benchmark the MLIR code.

    Returns:
        (execution time in ns, success)
    """
    if bufferize_transform_code is None:
        bufferize_transform_code = BUFFERIZATION_AND_LOWER_V_TRANSFORM_CODE
    if pass_pipeline is None:
        pass_pipeline = PASS_PIPELINE

    bufferized = run_transform_code(code, bufferize_transform_code)
    return execute_bufferized_code(
        bufferized,
        pass_pipeline
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
