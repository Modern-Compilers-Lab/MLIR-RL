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


def lower(module: Module, pass_file: str):
    with open(pass_file) as f:
        pipeline = f.read()

    pm = PassManager.parse(pipeline, module.context)
    pm.run(module.operation)
