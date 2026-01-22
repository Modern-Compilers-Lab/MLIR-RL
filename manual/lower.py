import sys
from mlir._mlir_libs._mlir.ir import Context, Module  # type: ignore
from mlir.passmanager import PassManager
from mlir.dialects.transform import interpreter


def main():
    pass_pipeline = """builtin.module(
        canonicalize,
        buffer-deallocation-pipeline,
        convert-bufferization-to-memref,
        convert-linalg-to-loops,
        loop-invariant-code-motion,
        scf-forall-to-parallel,
        convert-scf-to-openmp,
        expand-strided-metadata,
        finalize-memref-to-llvm,
        convert-scf-to-cf,
        lower-affine,

        convert-openmp-to-llvm,
        convert-vector-to-llvm,
        convert-math-to-llvm,
        convert-math-to-libm,
        finalize-memref-to-llvm,
        mem2reg,
        convert-func-to-llvm,
        convert-index-to-llvm,
        convert-arith-to-llvm,
        convert-cf-to-llvm,

        reconcile-unrealized-casts,
        canonicalize,
        cse
    )"""

    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            code = f.read()
    else:
        code = sys.stdin.read()

    with Context():
        module = Module.parse(code)
        pm = PassManager.parse(pass_pipeline)

    bufferize(module)
    pm.run(module.operation)

    print(module)


def bufferize(module: Module):
    """Apply bufferization

    Args:
        module: The MLIR module to transform.
    """
    transform_code = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
            transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
            %empty = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.op<"tensor.empty">
            transform.bufferization.empty_tensor_to_alloc_tensor %empty : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">

            transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op

            transform.yield
        }
    }"""

    t_module = Module.parse(transform_code, module.context)
    interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)


if __name__ == "__main__":
    main()
