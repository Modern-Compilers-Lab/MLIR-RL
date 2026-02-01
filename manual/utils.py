from mlir._mlir_libs._mlir.ir import Module  # type: ignore
from mlir.passmanager import PassManager


def bufferize(module: Module):
    pass_pipeline = """builtin.module(
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

    pm = PassManager.parse(pass_pipeline, module.context)
    pm.run(module.operation)


def main_lower(module: Module):
    pass_pipeline = """builtin.module(
        canonicalize, cse,
        func.func(
            promote-buffers-to-stack{max-alloc-size-in-bytes=32768}
        ),
        convert-linalg-to-loops,
        loop-invariant-code-motion,
        scf-forall-to-parallel,
        convert-scf-to-openmp,
        convert-openmp-to-llvm,
        expand-strided-metadata,
        lower-affine,
        convert-scf-to-cf,

        convert-ub-to-llvm,
        convert-vector-to-llvm{enable-x86vector},
        convert-math-to-llvm,
        convert-math-to-libm,
        finalize-memref-to-llvm,
        convert-func-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,

        reconcile-unrealized-casts,
        canonicalize, cse
    )"""

    pm = PassManager.parse(pass_pipeline, module.context)
    pm.run(module.operation)


def llm_lower(module: Module):
    pass_pipeline = """builtin.module(
        canonicalize, cse,
        func.func(
            promote-buffers-to-stack{max-alloc-size-in-bytes=32768}
        ),
        canonicalize, cse,
        convert-linalg-to-loops,
        lower-affine,
        canonicalize, cse,
        convert-scf-to-openmp,
        canonicalize, cse,
        convert-openmp-to-llvm,
        canonicalize, cse,
        expand-strided-metadata,
        lower-affine,
        canonicalize, cse,
        canonicalize, cse,
        lower-affine,
        canonicalize, cse,
        convert-scf-to-cf,
        canonicalize, cse,
        convert-math-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-vector-to-llvm{enable-x86vector},
        convert-ub-to-llvm,
        finalize-memref-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,
        convert-func-to-llvm,
        reconcile-unrealized-casts,
        canonicalize, cse
    )"""

    pm = PassManager.parse(pass_pipeline, module.context)
    pm.run(module.operation)


lower = main_lower
