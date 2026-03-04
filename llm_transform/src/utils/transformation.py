import contextlib
import ctypes
import os
from pathlib import Path
import subprocess
import tempfile

from mlir.ir import Context, Module
from mlir.passmanager import PassManager
from mlir.dialects.transform import interpreter

PARENT_DIR = Path(__file__).parents[2]


def bufferize_module(module: Module):
    bufferize_pipeline = """builtin.module(
        eliminate-empty-tensors,
        empty-tensor-to-alloc-tensor,
        one-shot-bufferize{
            bufferize-function-boundaries
            unknown-type-conversion=identity-layout-map
            function-boundary-type-conversion=identity-layout-map
        },
        buffer-results-to-out-params{hoist-static-allocs add-result-attr},
        canonicalize, cse
    )"""
    apply_pipeline_to_module(module, bufferize_pipeline)


def apply_pipeline_to_module(module: Module, pass_pipeline: str):
    pm = PassManager.parse(pass_pipeline, module.context)
    pm.run(module.operation)


def transform_module(module: Module, transform_schedule: str):
    t_module = Module.parse(transform_schedule, module.context)
    interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)


def transform_and_lower(id: int, transform_schedule: str, mlir_passes: str, llvm_passes, llvm_flags: str, llc_flags: str, bufferize_first: bool):
    results: dict[str, str] = {}

    with open(PARENT_DIR / 'data' / 'matmul' / f'{id}.mlir', 'r') as f:
        code = f.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)

    if bufferize_first:
        bufferize_module(module)

    transform_module(module, transform_schedule)
    results['mlir_transformed'] = str(module)

    if not bufferize_first:
        bufferize_module(module)

    apply_pipeline_to_module(module, mlir_passes)

    with compile_aot(str(module), llvm_passes, llvm_flags, llc_flags, intermediate_outs=results):
        pass

    return results


@contextlib.contextmanager
def compile_aot(mlir_code: str, llvm_passes: str, llvm_flags: str, llc_flags: str, intermediate_outs: dict | None = None):
    """Compile MLIR to shared lib via opt+llc and run via ctypes (no JIT)."""
    conda_lib = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    so_file = tempfile.NamedTemporaryFile(suffix=".so")

    # MLIR → LLVM IR
    result_llvm = subprocess.run(
        ["mlir-translate", "--mlir-to-llvmir"],
        input=mlir_code, text=True,
        check=True, capture_output=True
    )
    if intermediate_outs is not None:
        intermediate_outs['llvm'] = result_llvm.stdout

    # Optimize with opt (llvm_flags go to opt as CL options)
    opt_cmd = ["opt", "--mcpu=native", f"--passes={llvm_passes}", "-S"]
    if llvm_flags:
        for flag in llvm_flags.split(','):
            flag = flag.strip()
            if flag:
                opt_cmd.insert(1, f"--{flag}")
    result_opt = subprocess.run(
        opt_cmd, input=result_llvm.stdout,
        text=True, check=True, capture_output=True
    )
    if intermediate_outs is not None:
        intermediate_outs['llvm_opt'] = result_opt.stdout

    # Compile to object file (llc gets codegen-specific flags)
    llc_cmd = [
        "llc", "-filetype=obj", "-relocation-model=pic",
        "--mcpu=native", "-O3"
    ]
    if llc_flags:
        for flag in llc_flags.split(','):
            flag = flag.strip()
            if flag:
                llc_cmd.insert(1, f"--{flag}")
    result_llc = subprocess.run(
        llc_cmd, input=result_opt.stdout,
        text=True, check=True, capture_output=True
    )

    # Compile to assembly for analysis
    asm_cmd = llc_cmd.copy()
    asm_cmd[next(i for i, arg in enumerate(asm_cmd) if arg == "-filetype=obj")] = "-filetype=asm"
    result_asm = subprocess.run(
        asm_cmd, input=result_opt.stdout,
        text=True, check=True, capture_output=True
    )
    if intermediate_outs is not None:
        intermediate_outs['asm'] = result_asm.stdout

    # Link to shared library
    subprocess.run([
        "gcc", "-shared", f"-L{conda_lib}", "-lomp",
        "-lmlir_runner_utils", "-lmlir_c_runner_utils",
        "-lm", f"-Wl,-rpath,{conda_lib}", "-o", so_file.name
    ], input=result_llc.stdout, text=True, check=True)

    # Load and call via _mlir_ciface_main(ptr A, ptr B, ptr C) -> i64
    lib = ctypes.CDLL(so_file.name)
    func = lib._mlir_ciface_main
    func.restype = ctypes.c_int64

    try:
        yield func
    finally:
        so_file.close()
