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


def transform_and_lower(id: str, transform_schedule: str, mlir_passes: str, llvm_passes, llvm_flags: str, llc_flags: str, bufferize_first: bool):
    results: dict[str, str] = {}

    out_dir = PARENT_DIR / 'out' / id
    out_dir.mkdir(parents=True, exist_ok=True)

    name, instance = id.rsplit("_", 1)
    mlir_path = PARENT_DIR / 'data' / name / f'{instance}.mlir'
    if not mlir_path.exists():
        raise FileNotFoundError(f"MLIR source not found: {mlir_path}")
    with open(mlir_path, 'r') as f:
        code = f.read()

    with Context() as ctx:
        ctx.load_all_available_dialects()
        module = Module.parse(code)

    if bufferize_first:
        bufferize_module(module)

    transform_module(module, transform_schedule)
    mlir_transformed_path = out_dir / 'transformed.mlir'
    mlir_transformed_path.write_text(str(module))
    results['mlir_transformed'] = str(mlir_transformed_path)

    if not bufferize_first:
        bufferize_module(module)

    apply_pipeline_to_module(module, mlir_passes)

    with compile_aot(str(module), llvm_passes, llvm_flags, llc_flags, out_dir=out_dir, intermediate_outs=results):
        pass

    return results


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


@contextlib.contextmanager
def compile_aot(mlir_code: str, llvm_passes: str, llvm_flags: str, llc_flags: str, out_dir: Path | None = None, intermediate_outs: dict | None = None):
    """Compile MLIR to shared lib via opt+llc and run via ctypes (no JIT)."""
    if 'CONDA_PREFIX' not in os.environ:
        raise EnvironmentError("No Conda environment detected. Please activate a Conda environment before running this function.")
    conda_lib = os.path.join(os.environ['CONDA_PREFIX'], 'lib')
    tmp_dir = PARENT_DIR / "tmp"
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False, dir=tmp_dir) as obj_file, \
         tempfile.NamedTemporaryFile(suffix=".so", delete=False, dir=tmp_dir) as so_file:
        pass

    try:
        # MLIR → LLVM IR
        result_llvm = subprocess.run(
            ["mlir-translate", "--mlir-to-llvmir"],
            input=mlir_code, text=True,
            capture_output=True
        )
        if result_llvm.returncode != 0:
            raise RuntimeError(f"mlir-translate failed:\n{result_llvm.stderr}")
        if intermediate_outs is not None and out_dir is not None:
            llvm_path = out_dir / 'llvm.ll'
            llvm_path.write_text(result_llvm.stdout)
            intermediate_outs['llvm'] = str(llvm_path)

        # Optimize with opt (llvm_flags go to opt as CL options)
        opt_cmd = ["opt", "--mcpu=native", f"--passes={llvm_passes}", "-S"]
        if llvm_flags:
            for flag in llvm_flags.split(','):
                flag = flag.strip()
                if flag:
                    opt_cmd.insert(1, f"--{flag}")
        result_opt = subprocess.run(
            opt_cmd, input=result_llvm.stdout,
            text=True, capture_output=True
        )
        if result_opt.returncode != 0:
            raise RuntimeError(f"opt failed:\n{result_opt.stderr}")
        if intermediate_outs is not None and out_dir is not None:
            llvm_opt_path = out_dir / 'llvm_opt.ll'
            llvm_opt_path.write_text(result_opt.stdout)
            intermediate_outs['llvm_opt'] = str(llvm_opt_path)

        # Compile to object file (llc gets codegen-specific flags)
        llc_cmd = ["llc", "-relocation-model=pic", "-mcpu=native", "-O3"]
        if llc_flags:
            for flag in llc_flags.split(','):
                flag = flag.strip()
                if flag:
                    llc_cmd.insert(1, f"--{flag}")
        result_obj = subprocess.run(
            llc_cmd + ["-filetype=obj", "-o", obj_file.name],
            input=result_opt.stdout, text=True, capture_output=True
        )
        if result_obj.returncode != 0:
            raise RuntimeError(f"llc (obj) failed:\n{result_obj.stderr}")

        # Compile to assembly for analysis
        result_asm = subprocess.run(
            llc_cmd + ["-filetype=asm"], input=result_opt.stdout,
            text=True, capture_output=True
        )
        if result_asm.returncode != 0:
            raise RuntimeError(f"llc (asm) failed:\n{result_asm.stderr}")
        if intermediate_outs is not None and out_dir is not None:
            asm_path = out_dir / 'asm.s'
            asm_path.write_text(result_asm.stdout)
            intermediate_outs['asm'] = str(asm_path)

        # Link to shared library
        result_link = subprocess.run([
            "gcc", "-shared", obj_file.name,
            f"-L{conda_lib}", "-lomp",
            "-lmlir_runner_utils", "-lmlir_c_runner_utils",
            "-lm", f"-Wl,-rpath,{conda_lib}", "-o", so_file.name
        ], text=True, capture_output=True)
        if result_link.returncode != 0:
            raise RuntimeError(f"gcc linking failed:\n{result_link.stderr}")

        # Load and call via _mlir_ciface_main(ptr A, ptr B, ptr C) -> i64
        lib = ctypes.CDLL(so_file.name)
        func = lib._mlir_ciface_main
        func.restype = ctypes.c_int64

        yield func
    finally:
        Path(obj_file.name).unlink(missing_ok=True)
        Path(so_file.name).unlink(missing_ok=True)
