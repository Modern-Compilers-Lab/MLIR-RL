#!/usr/bin/env python
"""Run the array-dataflow equivalence verifier on a single MLIR file.

The input is a kernel + transform schedule (like `tests/validation/*.mlir`): one
`func.func` plus a trailing `module attributes {transform.with_named_sequence}`.
The kernel is taken as the *original*; the schedule is applied to a copy to
produce the *transformed* function.

The script tags, lowers and combines the two functions and runs
`check-array-dataflow-equivalence`, printing the verdict on stdout:

  valid    - the transform preserves the original array dataflow
  invalid  - a dependence is not preserved (the transform is illegal)

The verdict is always printed to stdout and the process exits 0. Anything else
(plugins not built, malformed input, the schedule failing to apply or lower) is
an error: it is reported on stderr and the process exits 1. With `--verbose` the
verifier's trace and diagnostics are forwarded to stderr (still exit 0 for a
clean verdict — the exit code distinguishes verbose output from a real error).

    python verify_equivalence.py kernel_and_schedule.mlir
    python verify_equivalence.py input.mlir --verbose
"""

import argparse
import contextlib
import ctypes
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
LIB = HERE / "build" / "lib"
PLUGINS = ("libTagLinalgOps.so", "libRaiseSCFToAffine.so", "libEquivalenceVerifier.so")


def load_plugins():
    """Dlopen each plugin so its pass self-registers; raise if any can't load."""
    for name in PLUGINS:
        path = LIB / name
        try:
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            raise RuntimeError(
                f"could not load plugin {path}: {exc}; "
                "follow documentation to build it"
            ) from exc


# The plugins must self-register before the MLIR bindings are imported below.
try:
    load_plugins()
except Exception as exc:
    print(f"error: {exc}", file=sys.stderr)
    sys.exit(1)

from mlir.ir import Context, Module, StringAttr
from mlir.dialects.func import FuncOp
from mlir.passmanager import PassManager
from mlir.dialects.transform import interpreter

# Tag every linalg op (still in linalg form) with a stable `eq_id_<n>` so the
# verifier can line up an original access with its transformed counterpart.
TAG_PIPELINE = "builtin.module(func.func(tag-linalg-ops-for-equivalence))"
# Lower a tagged kernel to the affine + memref form the verifier consumes.
LOWER_PIPELINE = (
    "builtin.module("
    "func.func(scf-forall-to-for,raise-scf-to-affine),"
    "convert-linalg-to-affine-loops,"
    "func.func(fold-memref-alias-ops,affine-raise-from-memref))"
)
CHECK_PIPELINE = (
    "builtin.module(check-array-dataflow-equivalence"
    "{{original-func=original transformed-func=transformed{verbose}}})"
)
# Bufferize a tensor-form module to memrefs (no-op on already-memref input).
BUFFERIZE_PIPELINE = (
    "builtin.module("
    "eliminate-empty-tensors,"
    "empty-tensor-to-alloc-tensor,"
    "one-shot-bufferize{bufferize-function-boundaries "
    "unknown-type-conversion=identity-layout-map "
    "function-boundary-type-conversion=identity-layout-map},"
    "buffer-results-to-out-params{hoist-static-allocs add-result-attr},"
    "canonicalize,cse)"
)


def apply_pipeline_to_module(module: Module, pass_pipeline: str):
    PassManager.parse(pass_pipeline, module.context).run(module.operation)


def bufferize_module(module: Module):
    apply_pipeline_to_module(module, BUFFERIZE_PIPELINE)


def transform_module(module: Module, transform_schedule: str):
    t_module = Module.parse(transform_schedule, module.context)
    interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)


@contextlib.contextmanager
def capture_fds():
    """Redirect fds 1 and 2 to a temp file; append the captured text on exit."""
    sys.stdout.flush()
    sys.stderr.flush()
    old_out, old_err = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    collected: list[str] = []
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        yield collected
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_out, 1)
        os.dup2(old_err, 2)
        os.close(old_out)
        os.close(old_err)
        tmp.seek(0)
        collected.append(tmp.read().decode("utf-8", errors="replace"))
        tmp.close()


def split_kernel_and_schedule(code: str) -> tuple[str, str | None]:
    """Return (kernel_code, transform_schedule); schedule is None if absent."""
    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)
        schedule: str | None = None
        for op in list(module.body.operations):
            if op.operation.name != "builtin.module":
                continue
            if "transform.with_named_sequence" not in op.operation.attributes:
                continue
            schedule = str(op)
            op.operation.erase()
        return str(module), schedule


def append_func_as(dest: Module, src: Module, new_name: str):
    """Clone the first func.func from `src` into `dest`, renamed to `new_name`."""
    for op in src.body.operations:
        if isinstance(op, FuncOp):
            clone = op.operation.clone()
            clone.attributes["sym_name"] = StringAttr.get(new_name, dest.context)
            dest.body.append(clone)
            return
    raise ValueError("no func.func found to combine for the equivalence check")


def build_from_schedule(kernel_code: str, schedule: str) -> Module:
    """Tag, transform, lower and combine into one `original`/`transformed` module."""
    original = Module.parse(kernel_code)
    apply_pipeline_to_module(original, TAG_PIPELINE)  # tag before transforming
    transformed = original.operation.clone()

    bufferize_module(original)
    apply_pipeline_to_module(original, LOWER_PIPELINE)

    transform_module(transformed, schedule)
    bufferize_module(transformed)
    apply_pipeline_to_module(transformed, LOWER_PIPELINE)

    combined = Module.parse("module {}")
    append_func_as(combined, original, "original")
    append_func_as(combined, transformed, "transformed")
    return combined


def verify(path: Path, verbose: bool):
    """Print `valid`/`invalid` to stdout. Raise on any malformed input / failure."""
    code = path.read_text()
    kernel_code, schedule = split_kernel_and_schedule(code)
    if schedule is None:
        raise ValueError(
            "no transform schedule (module attributes "
            "{transform.with_named_sequence}) found in the input"
        )

    with Context() as ctx:
        ctx.load_all_available_dialects()
        combined = build_from_schedule(kernel_code, schedule)

        # The verifier emits its trace to stdout and its violation diagnostic to
        # stderr (and fails the pass) when a dependence is not preserved. Capture
        # both so the verdict on stdout stays clean; forward them in verbose mode.
        check = CHECK_PIPELINE.format(verbose=" verbose" if verbose else "")
        with capture_fds() as cap:
            failed = False
            try:
                apply_pipeline_to_module(combined, check)
            except Exception:
                failed = True
        trace = "\n".join(cap).strip()

    print("invalid" if failed else "valid")
    if verbose and trace:
        print(trace, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="MLIR file to verify")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="forward the verifier trace and diagnostics to stderr")
    args = ap.parse_args()

    try:
        if not args.input.is_file():
            raise FileNotFoundError(f"no such file: {args.input}")
        verify(args.input, args.verbose)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
