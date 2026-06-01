"""
Validation test harness.

For each MLIR file in `tests/validation/`:
  1. Split the file into a kernel and a transform schedule.
  2. Produce two variants: (A) kernel with schedule stripped, (B) schedule
     applied to the kernel and then stripped.
  3. Bufferize and lower both to a shared library.
  4. Generate random inputs, run both with matching copies, and compare
     the (in-place mutated) outputs.
  5. Run up to three independent detectors and watch their stderr:
       - mlir         : stderr emitted while applying the schedule and lowering.
       - legality     : the PolyhedralLegalityCheck plugin on the transformed
                        module.
       - equivalence  : the array-dataflow EquivalenceVerifier comparing the
                        original and transformed kernels (requires the
                        TagLinalgOps pass to run on both first).
     The (in-place mutated) outputs are the ground truth: if they differ the
     transform is illegal and at least one active detector must flag it; if
     they match no active detector may flag it. `--tool` restricts evaluation
     to a single detector.
"""

import argparse
import contextlib
import ctypes
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


GREEN = "32"
RED = "31"
YELLOW = "33"
CYAN = "36"
BOLD = "1"
FAINT = "2"


PARENT_DIR = Path(__file__).parent
TESTS_DIR = PARENT_DIR / "tests" / "validation"
PASSES_FILE = TESTS_DIR / "lowering_passes.txt"
LEGALITY_PLUGIN = (
    PARENT_DIR /
    "llm_transform" / "tools" / "c" / "dependence" /
    "build" / "lib" / "libPolyhedralLegalityCheck.so"
)
LEGALITY_PASS_PIPELINE = "builtin.module(convert-linalg-to-affine-loops,func.func(fold-memref-alias-ops,affine-raise-from-memref,check-polyhedral-legality))"

EQUIVALENCE_DIR = (
    PARENT_DIR / "llm_transform" / "tools" / "c" / "equivalence" / "build" / "lib"
)
TAG_LINALG_PLUGIN = EQUIVALENCE_DIR / "libTagLinalgOps.so"
EQUIVALENCE_PLUGIN = EQUIVALENCE_DIR / "libEquivalenceVerifier.so"
RAISE_SCF_PLUGIN = EQUIVALENCE_DIR / "libRaiseSCFToAffine.so"

# Tag every linalg op (on the still-linalg form) with a stable `eq_id_<n>` so
# the verifier can line up an original access with its transformed counterpart.
TAG_LINALG_PIPELINE = "builtin.module(func.func(tag-linalg-ops-for-equivalence))"
# Lower a tagged kernel to the affine + memref form the verifier consumes.
# Tiling via the transform dialect emits `scf.forall`/`scf.for` tile loops; we
# normalize those to `affine.for` (scf-forall-to-for + raise-scf-to-affine)
# *before* lowering the linalg body, otherwise convert-linalg-to-affine-loops
# would build inner affine loops bounded by SCF induction variables — not a
# legal affine quantity. The tail mirrors the legality plugin's preprocessing.
EQUIVALENCE_LOWER_PIPELINE = "builtin.module(func.func(scf-forall-to-for,raise-scf-to-affine),convert-linalg-to-affine-loops,func.func(fold-memref-alias-ops,affine-raise-from-memref))"
# Compare the `original` and `transformed` functions placed in a single module.
EQUIVALENCE_CHECK_PIPELINE = "builtin.module(check-array-dataflow-equivalence{original-func=original transformed-func=transformed})"

ALL_TOOLS = ("mlir", "legality", "equivalence")
TOOL_LABELS = {"mlir": "MLIR", "legality": "Legality", "equivalence": "Equivalence"}

def _prefix_tool_name(message: str, tool: str) -> str:
    prefix = f"[{TOOL_LABELS[tool]}] "
    return "\n".join(prefix + line for line in message.splitlines())


def load_legality_plugin() -> bool:
    """Dlopen the PolyhedralLegalityCheck plugin so its pass self-registers.

    Returns True on success. If the .so is missing, prints a hint and returns
    False so the plugin-detection column is reported as n/a.
    """
    path = Path(os.environ.get("LEGALITY_PLUGIN", LEGALITY_PLUGIN))
    if not path.exists():
        print(
            f"[warn] legality plugin not found at {path}; "
            f"build it with `make -C llm_transform/tools/c/dependence "
            f"PREFIX=$(python -c 'import sys;print(sys.prefix)')` "
            f"or set LEGALITY_PLUGIN=/path/to/libPolyhedralLegalityCheck.so",
            file=sys.stderr,
        )
        return False
    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return True


def load_equivalence_plugins() -> bool:
    """Dlopen the TagLinalgOps, RaiseSCFToAffine and EquivalenceVerifier plugins
    so their passes self-register.

    All of `tag-linalg-ops-for-equivalence`, `raise-scf-to-affine` and
    `check-array-dataflow-equivalence` are needed; returns True only if every .so
    loads. If any is missing, prints a build hint and returns False so the
    equivalence column is n/a.
    """
    ok = True
    for env, default, pass_name in (
        ("TAG_LINALG_PLUGIN", TAG_LINALG_PLUGIN, "tag-linalg-ops-for-equivalence"),
        ("RAISE_SCF_PLUGIN", RAISE_SCF_PLUGIN, "raise-scf-to-affine"),
        ("EQUIVALENCE_PLUGIN", EQUIVALENCE_PLUGIN, "check-array-dataflow-equivalence"),
    ):
        path = Path(os.environ.get(env, default))
        if not path.exists():
            print(
                f"[warn] {pass_name} plugin not found at {path}; "
                f"build it with `make -C llm_transform/tools/c/equivalence "
                f"PREFIX=$(python -c 'import sys;print(sys.prefix)')` "
                f"or set {env}=/path/to/plugin.so",
                file=sys.stderr,
            )
            ok = False
            continue
        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
    return ok


LEGALITY_LOADED = load_legality_plugin()
EQUIVALENCE_LOADED = load_equivalence_plugins()


from mlir.ir import (
    Context,
    F32Type,
    F64Type,
    IntegerType,
    MemRefType,
    Module,
    StringAttr,
    UnitAttr,
)
from mlir.dialects.func import FuncOp
from mlir.execution_engine import ExecutionEngine
from mlir.runtime import get_ranked_memref_descriptor
from llm_transform.utils.transformation import (
    apply_pipeline_to_module,
    apply_pipeline_to_module_with_opt,
    bufferize_module,
    transform_module,
    transform_module_with_opt,
)


@contextlib.contextmanager
def capture_stderr_fd():
    """Redirect fd 2 to a temp file; appends captured text to the yielded list."""
    sys.stderr.flush()
    old_fd = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    collected: list[str] = []
    try:
        os.dup2(tmp.fileno(), 2)
        yield collected
    finally:
        sys.stderr.flush()
        os.dup2(old_fd, 2)
        os.close(old_fd)
        tmp.seek(0)
        collected.append(tmp.read().decode("utf-8", errors="replace"))
        tmp.close()


def split_test(code: str) -> tuple[str, str | None]:
    """Parse the file and return (kernel_code, transform_schedule).

    The transform schedule is the inner `module attributes {transform.with_named_sequence}`.
    """
    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)
        transform_str: str | None = None
        for op in list(module.body.operations):
            if op.operation.name != "builtin.module":
                continue
            attrs = op.operation.attributes
            try:
                _ = attrs["transform.with_named_sequence"]
            except KeyError:
                continue
            transform_str = str(op)
            op.operation.erase()
        kernel_code = str(module)
    return kernel_code, transform_str


def kernel_func_info(module: Module) -> tuple[str, list]:
    """Return (func_name, list_of_memref_input_types) for the first func.func."""
    for op in module.body.operations:
        if isinstance(op, FuncOp):
            return op.name.value, list(op.type.inputs)
    raise ValueError("No func.func found in kernel module")


def memref_dtype(t: MemRefType):
    et = t.element_type
    if isinstance(et, F32Type):
        return np.float32
    if isinstance(et, F64Type):
        return np.float64
    if isinstance(et, IntegerType):
        if et.width == 32:
            return np.int32
        if et.width == 64:
            return np.int64
    raise ValueError(f"unsupported element type {et}")


def gen_inputs(input_types, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    arrs = []
    for t in input_types:
        if not isinstance(t, MemRefType):
            raise ValueError(f"expected memref arg, got {t}")
        dtype = memref_dtype(t)
        if np.issubdtype(dtype, np.floating):
            arr = rng.standard_normal(t.shape).astype(dtype)
        else:
            arr = rng.integers(0, 100, size=t.shape, dtype=dtype)
        arrs.append(np.ascontiguousarray(arr))
    return arrs


def _mark_c_interface(module: Module):
    """Ensure every top-level func has llvm.emit_c_interface so we can call it via ctypes."""
    for op in module.body.operations:
        if isinstance(op, FuncOp):
            op.operation.attributes["llvm.emit_c_interface"] = UnitAttr.get(op.context)


def prepare_module_for_lowering(module: Module):
    """Mark the main kernel function(s) with llvm.emit_c_interface and bufferize the module."""
    _mark_c_interface(module)
    bufferize_module(module)


def lower_to_llvm_mlir(module: Module):
    passes = PASSES_FILE.read_text()
    apply_pipeline_to_module_with_opt(module, passes)


def _runtime_shared_libs() -> list[str]:
    """Locate the MLIR runner utility libraries the ExecutionEngine needs."""
    conda_lib = Path(os.environ["CONDA_PREFIX"]) / "lib"
    libs = []
    for name in ("libmlir_runner_utils.so", "libmlir_c_runner_utils.so", "libomp.so"):
        p = conda_lib / name
        if p.exists():
            libs.append(str(p))
    return libs


def build_execution_engine(module: Module) -> ExecutionEngine:
    """JIT-compile the LLVM-dialect module with MLIR's ExecutionEngine."""
    return ExecutionEngine(
        module,
        opt_level=3,
        shared_libs=_runtime_shared_libs(),
    )


def run_legality_plugin(module: Module) -> str:
    """Apply the PolyhedralLegalityCheck pass to `code`.

    Returns the captured stderr output from the plugin
    """
    if not LEGALITY_LOADED:
        return ''
    module_clone: Module = module.operation.clone()
    with capture_stderr_fd() as cap:
        try:
            apply_pipeline_to_module(module_clone, LEGALITY_PASS_PIPELINE)
        except Exception as exc:
            cap.append(f"[legality exception] {exc}")
    return "\n".join(cap).strip()


def _append_func_as(dest: Module, src: Module, new_name: str):
    """Clone the first func.func from `src` into `dest`, renamed to `new_name`.

    `src` and `dest` must share a context. The verifier expects both functions in
    a single module under the names `original` and `transformed`; renaming also
    avoids the symbol collision that would otherwise occur (both kernels carry the
    same original name).
    """
    for op in src.body.operations:
        if isinstance(op, FuncOp):
            clone = op.operation.clone()
            clone.attributes["sym_name"] = StringAttr.get(new_name, dest.context)
            dest.body.append(clone)
            return
    raise ValueError("no func.func found to combine for equivalence check")


def run_equivalence_verifier(kernel_code: str, transform_sched: str) -> str:
    """Prove array-dataflow equivalence of the original and transformed kernels.

    Builds two affine functions from `kernel_code` — one untouched (`original`),
    one with `transform_sched` applied (`transformed`) — tagging the linalg ops of
    each with stable `eq_id_<n>` ids before lowering so the verifier can match an
    original access against its transformed counterpart. Returns the captured
    stderr from `check-array-dataflow-equivalence` (non-empty means a violation or
    a failed match was reported).
    """
    if not EQUIVALENCE_LOADED:
        return ''

    with Context() as ctx:
        ctx.load_all_available_dialects()
        # Build both functions to the tagged affine form. Any diagnostics from
        # the transform/bufferize/lowering belong to MLIR's column, not the
        # verifier, so swallow them here.
        with capture_stderr_fd():
            original = Module.parse(kernel_code)
            apply_pipeline_to_module(original, TAG_LINALG_PIPELINE)
            transformed = original.operation.clone()

            bufferize_module(original)
            apply_pipeline_to_module(original, EQUIVALENCE_LOWER_PIPELINE)

            transform_module(transformed, transform_sched)
            bufferize_module(transformed)
            apply_pipeline_to_module(transformed, EQUIVALENCE_LOWER_PIPELINE)

            combined = Module.parse("module {}")
            _append_func_as(combined, original, "original")
            _append_func_as(combined, transformed, "transformed")

    with capture_stderr_fd() as cap:
        try:
            apply_pipeline_to_module(combined, EQUIVALENCE_CHECK_PIPELINE)
        except Exception as exc:
            cap.append(f"[equivalence exception] {exc}")
    return "\n".join(cap).strip()


def run_kernel(engine: ExecutionEngine, func_name: str, arrs):
    """Invoke the JIT-compiled kernel via MLIR's ExecutionEngine."""
    args = [
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a)))
        for a in arrs
    ]
    engine.invoke(func_name, *args)


def run_test(path: Path, active_tools: set[str]) -> tuple[bool | None, dict[str, bool], str]:
    """Run the test and the requested detectors.

    Returns (outputs_equal, detections, message) where `detections` maps each
    active tool name to whether it flagged the transform, and `outputs_equal` is
    the ground-truth comparison of the executed kernels (None if the transform
    itself failed to apply).
    """

    # Read the code and split the kernel from the transform schedule
    code = path.read_text()
    kernel_code, transform_sched = split_test(code)
    if transform_sched is None:
        raise ValueError("no transform schedule found in test file")

    with Context() as ctx:
        ctx.load_all_available_dialects()
        base_module = Module.parse(kernel_code)
    prepare_module_for_lowering(base_module)

    # Build the transformed kernel (schedule applied).
    with Context() as ctx:
        ctx.load_all_available_dialects()
        trans_module = Module.parse(kernel_code)
    # If anything is captured here it means MLIR has detected an issue
    with capture_stderr_fd() as cap:
        try:
            transform_module_with_opt(trans_module, transform_sched)
            transform_ran = True
        except Exception as exc:
            cap.append(f"[transformation exception] {exc}")
            transform_ran = False

    mlir_stderr = "\n".join(cap).strip()
    if not transform_ran:
        # If the transformation itself failed, we consider that a form of detection.
        return None, {"mlir": bool(mlir_stderr)}, mlir_stderr

    # After the transformations are applied we can prepare the module for lowering
    prepare_module_for_lowering(trans_module)

    detections: dict[str, bool] = {}
    messages: list[str] = []

    # The "mlir" detector is a byproduct of applying the schedule above.
    if "mlir" in active_tools:
        detections["mlir"] = bool(mlir_stderr)
        if mlir_stderr:
            messages.append(mlir_stderr)

    # Independent detector: the PolyhedralLegalityCheck pass on the transformed
    # module, orthogonal to MLIR's own stderr-based detection during lowering.
    if "legality" in active_tools:
        legality_stderr = run_legality_plugin(trans_module)
        detections["legality"] = bool(legality_stderr)
        if legality_stderr:
            messages.append(legality_stderr)

    # Independent detector: the array-dataflow EquivalenceVerifier comparing the
    # original kernel against the transformed one (rebuilt from source so it can
    # be tagged on the still-linalg form before lowering to affine).
    if "equivalence" in active_tools:
        equivalence_stderr = run_equivalence_verifier(kernel_code, transform_sched)
        detections["equivalence"] = bool(equivalence_stderr)
        if equivalence_stderr:
            messages.append(equivalence_stderr)

    func_name, input_types = kernel_func_info(base_module)
    inputs_base = gen_inputs(input_types, seed=0xC0FFEE)
    inputs_trans = [a.copy() for a in inputs_base]

    # Baseline: kernel with no transform applied.
    lower_to_llvm_mlir(base_module)
    base_engine = build_execution_engine(base_module)

    # Transformed: kernel with schedule applied.
    lower_to_llvm_mlir(trans_module)
    trans_engine = build_execution_engine(trans_module)

    run_kernel(base_engine, func_name, inputs_base)
    run_kernel(trans_engine, func_name, inputs_trans)

    equal = all(
        np.allclose(a, b, rtol=1e-5, atol=1e-6)
        for a, b in zip(inputs_base, inputs_trans)
    )

    return equal, detections, "\n\n".join(messages)


def main():
    ap = argparse.ArgumentParser(description="Run MLIR dependence-violation validation tests.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print captured detector stderr for each test.")
    ap.add_argument("--filter", default=None, help="Only run tests whose filename contains this substring.")
    ap.add_argument(
        "--tool", choices=("all", *ALL_TOOLS), default="all",
        help="Only evaluate the given detection tool (default: all).",
    )
    args = ap.parse_args()

    active_tools = set(ALL_TOOLS) if args.tool == "all" else {args.tool}

    if not TESTS_DIR.is_dir():
        print(f"No tests directory at {TESTS_DIR}", file=sys.stderr)
        sys.exit(2)

    # Get all test cases and filter them
    tests = sorted(TESTS_DIR.glob("*.mlir"))
    if args.filter:
        tests = [t for t in tests if args.filter in t.name]
    if not tests:
        print(f"No .mlir tests found in {TESTS_DIR}", file=sys.stderr)
        sys.exit(2)

    passed = failed = 0
    for path in tests:
        print(f"{_c(CYAN, '[RUN ]')} {path.name}", flush=True)
        try:
            ok, detections, message = run_test(path, active_tools)
        except Exception as exc:
            print(f"{_c(RED + ';' + BOLD, '[FAIL]')} {path.name}: harness error: {exc}")
            failed += 1
            continue

        if ok is None:
            tag = _c(YELLOW + ';' + BOLD, "[UNKW]")
        else:
            any_detected = any(detections.values())
            test_passed = (ok and not any_detected) or (not ok and any_detected)
            if test_passed:
                tag = _c(GREEN + ';' + BOLD, "[PASS]")
                passed += 1
            else:
                tag = _c(RED + ';' + BOLD, "[FAIL]")
                failed += 1
        print(tag, path.name)

        outputs_str = _c(GREEN, "match") if ok else _c(RED, "differ")
        print("    - Outputs:", outputs_str)
        for tool in ALL_TOOLS:
            if tool not in detections:
                continue
            tool_str = _c(YELLOW, "detected") if detections[tool] else _c(FAINT, "silent")
            print(f"    - {TOOL_LABELS[tool]}:", tool_str)
        if args.verbose and message:
            print("    - Message:")
            prefix = "      | "
            print(_c(YELLOW, prefix + message.replace("\n", "\n" + prefix)))

    if failed == 0:
        summary = _c(GREEN + ';' + BOLD, f"{passed}/{passed + failed} passed, {failed} failed")
    else:
        summary = _c(RED + ';' + BOLD, f"{passed}/{passed + failed} passed, {failed} failed")
    print(f"\n{summary}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
