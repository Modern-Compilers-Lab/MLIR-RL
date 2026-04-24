"""
Validation test harness.

For each MLIR file in `tests/validation/`:
  1. Split the file into a kernel and a transform schedule.
  2. Produce two variants: (A) kernel with schedule stripped, (B) schedule
     applied to the kernel and then stripped.
  3. Bufferize and lower both to a shared library.
  4. Generate random inputs, run both with matching copies, and compare
     the (in-place mutated) outputs.
  5. Watch stderr during the entire pipeline. If outputs differ and MLIR
     was silent, the test fails; if MLIR emitted any warning/error, the
     test passes (detection succeeded).
"""

import argparse
import contextlib
import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PARENT_DIR = Path(__file__).parent
sys.path.insert(0, str(PARENT_DIR / "src" / "utils"))

from mlir.ir import (  # noqa: E402
    Context,
    F32Type,
    F64Type,
    IntegerType,
    MemRefType,
    Module,
    UnitAttr,
)
from mlir.dialects.func import FuncOp  # noqa: E402
from mlir.runtime import get_ranked_memref_descriptor  # noqa: E402
from transformation import (  # noqa: E402
    apply_pipeline_to_module,
    bufferize_module,
    transform_module,
)

TESTS_DIR = PARENT_DIR / "tests" / "validation"
PASSES_FILE = PARENT_DIR / "resources" / "base_passes.txt"
TMP_DIR = PARENT_DIR / "tmp"


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


def kernel_func_info(code: str) -> tuple[str, list]:
    """Return (func_name, list_of_memref_input_types) for the first func.func."""
    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)
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


def lower_to_llvm_mlir(kernel_code: str, stderr_sink: list[str]) -> str:
    with capture_stderr_fd() as cap:
        with Context() as ctx:
            ctx.load_all_available_dialects()
            module = Module.parse(kernel_code)
            _mark_c_interface(module)
            bufferize_module(module)
            passes = PASSES_FILE.read_text()
            apply_pipeline_to_module(module, passes)
            lowered = str(module)
    stderr_sink.extend(cap)
    return lowered


def compile_shared_lib(llvm_mlir: str, stderr_sink: list[str]) -> tuple[str, str]:
    """Compile the LLVM-dialect MLIR into a .so; return (so_path, obj_path)."""
    conda_lib = Path(os.environ["CONDA_PREFIX"]) / "lib"
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    obj = tempfile.NamedTemporaryFile(suffix=".o", delete=False, dir=TMP_DIR)
    so = tempfile.NamedTemporaryFile(suffix=".so", delete=False, dir=TMP_DIR)
    obj.close()
    so.close()

    def run(cmd, inp=None):
        r = subprocess.run(cmd, input=inp, text=True, capture_output=True)
        if r.stderr:
            stderr_sink.append(r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed (rc={r.returncode}): {r.stderr}")
        return r.stdout

    llvm_ir = run(["mlir-translate", "--mlir-to-llvmir"], inp=llvm_mlir)
    llvm_opt = run(["opt", "--mcpu=native", "--passes=default<O3>", "-S"], inp=llvm_ir)
    run(
        [
            "llc",
            "-relocation-model=pic",
            "-mcpu=native",
            "-O3",
            "-filetype=obj",
            "-o",
            obj.name,
        ],
        inp=llvm_opt,
    )
    run(
        [
            "gcc",
            "-shared",
            obj.name,
            f"-L{conda_lib}",
            "-lomp",
            "-lmlir_runner_utils",
            "-lmlir_c_runner_utils",
            "-lm",
            f"-Wl,-rpath,{conda_lib}",
            "-o",
            so.name,
        ]
    )
    return so.name, obj.name


def run_kernel(so_path: str, func_name: str, arrs, stderr_sink: list[str]):
    lib = ctypes.CDLL(so_path)
    fn = getattr(lib, f"_mlir_ciface_{func_name}")
    fn.restype = None
    args = [ctypes.pointer(get_ranked_memref_descriptor(a)) for a in arrs]
    with capture_stderr_fd() as cap:
        fn(*args)
    stderr_sink.extend(cap)


def run_test(path: Path, verbose: bool) -> tuple[bool, str, str]:
    code = path.read_text()
    kernel_code, transform_sched = split_test(code)
    if transform_sched is None:
        return False, "no transform schedule found", ""

    func_name, input_types = kernel_func_info(kernel_code)
    inputs_base = gen_inputs(input_types, seed=0xC0FFEE)
    inputs_trans = [a.copy() for a in inputs_base]

    mlir_stderr: list[str] = []

    # Build the transformed kernel (schedule applied).
    with capture_stderr_fd() as cap:
        try:
            with Context() as ctx:
                ctx.load_all_available_dialects()
                module = Module.parse(kernel_code)
                transform_module(module, transform_sched)
                transformed_code = str(module)
            transform_error = None
        except Exception as exc:
            transformed_code = None
            transform_error = str(exc)
    mlir_stderr.extend(cap)

    if transformed_code is None:
        combined = "".join(mlir_stderr) + f"\n[transform exception] {transform_error}"
        return True, "detected during transform", combined

    base_so = trans_so = base_obj = trans_obj = None
    try:
        # Baseline: kernel with no transform applied.
        base_llvm = lower_to_llvm_mlir(kernel_code, mlir_stderr)
        base_so, base_obj = compile_shared_lib(base_llvm, mlir_stderr)

        # Transformed: kernel with schedule applied.
        try:
            trans_llvm = lower_to_llvm_mlir(transformed_code, mlir_stderr)
            trans_so, trans_obj = compile_shared_lib(trans_llvm, mlir_stderr)
        except Exception as exc:
            return True, "detected during lowering/compile", "".join(mlir_stderr) + f"\n[compile exception] {exc}"

        run_kernel(base_so, func_name, inputs_base, mlir_stderr)
        run_kernel(trans_so, func_name, inputs_trans, mlir_stderr)

        equal = all(
            np.allclose(a, b, rtol=1e-5, atol=1e-6)
            for a, b in zip(inputs_base, inputs_trans)
        )
        mlir_complained = any(s.strip() for s in mlir_stderr)

        combined = "".join(mlir_stderr)
        if equal:
            return True, "outputs match", combined
        if mlir_complained:
            return True, "outputs differ, MLIR reported an issue", combined
        return False, "outputs differ, MLIR was silent", combined
    finally:
        for p in (base_so, base_obj, trans_so, trans_obj):
            if p:
                Path(p).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Run MLIR dependence-violation validation tests.")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print captured MLIR stderr for each test.")
    ap.add_argument("--filter", default=None, help="Only run tests whose filename contains this substring.")
    args = ap.parse_args()

    if not TESTS_DIR.is_dir():
        print(f"No tests directory at {TESTS_DIR}", file=sys.stderr)
        sys.exit(2)

    tests = sorted(TESTS_DIR.glob("*.mlir"))
    if args.filter:
        tests = [t for t in tests if args.filter in t.name]
    if not tests:
        print(f"No .mlir tests found in {TESTS_DIR}", file=sys.stderr)
        sys.exit(2)

    passed = failed = 0
    for path in tests:
        print(f"[RUN ] {path.name}", flush=True)
        try:
            ok, reason, captured = run_test(path, verbose=args.verbose)
        except Exception as exc:
            print(f"[FAIL] {path.name}: harness error: {exc}")
            failed += 1
            continue

        if ok:
            print(f"[PASS] {path.name}: {reason}")
            passed += 1
        else:
            print(f"[FAIL] {path.name}: {reason}")
            failed += 1

        if args.verbose and captured.strip():
            prefix = "    | "
            print(prefix + captured.rstrip().replace("\n", "\n" + prefix))

    total = passed + failed
    print(f"\n{passed}/{total} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
