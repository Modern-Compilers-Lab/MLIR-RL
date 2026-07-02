"""MLIR transformation passes for loop optimization.

This module provides functions for applying various loop transformation passes to MLIR code,
including tiling, interchange, parallelization, vectorization, and fusion. It interfaces
with the MLIR transform dialect for specifying and applying transformations.
"""

import os
import subprocess
from mlir._mlir_libs._mlir.ir import Module  # type: ignore
from mlir.dialects.transform import interpreter
from rl_autoschedular_paper_transformer.utils.bindings_process import BindingsProcess


def transform_TP(module: Module, operation_tag: str, tiling_sizes: list[int]):
    """Apply tiling and parallelization transformation to an operation.

    Tiles loops using forall constructs for parallelization.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the operation to transform.
        tiling_sizes: List of tiling factors for each loop.
    """
    # If tiling sizes are all zeros, means no tiling is needed
    if all([a == 0 for a in tiling_sizes]):
        return

    # Add full transform dialect code into the main code
    transform_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %op_tiled_{operation_tag}, %forall_{operation_tag} = transform.structured.tile_using_forall %op_{operation_tag} tile_sizes {str(tiling_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}'
    )

    __run_transform_code_wrapper(module, transform_code)


def transform_tile(module: Module, operation_tag: str, tiling_sizes: list[int]):
    """Apply tiling transformation to an operation using for loops.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the operation to transform.
        tiling_sizes: List of tiling factors for each loop.
    """
    # If tiling sizes are all zeros, means no tiling is needed
    if all([a == 0 for a in tiling_sizes]):
        return

    n_loops = sum([s != 0 for s in tiling_sizes])
    r = ', '.join(['!transform.any_op'] * n_loops)
    assert n_loops > 0, "No loops to tile"

    transform_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %tiled_op_{operation_tag}, %loops:{n_loops} = transform.structured.tile_using_for %op_{operation_tag} tile_sizes {str(tiling_sizes)} : (!transform.any_op) -> (!transform.any_op, {r})\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}\n'
    )

    __run_transform_code_wrapper(module, transform_code)


def transform_interchange(module: Module, operation_tag: str, interchange_list: list[int]):
    """Apply loop interchange transformation to an operation.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the operation to transform.
        interchange_list: Permutation of loop indices defining the new loop order.
    """
    # If the permutation list is same as the identity permutation, means no interchange is needed
    if interchange_list == list(range(len(interchange_list))):
        return

    transform_code = (
        f'module attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %gen_op_{operation_tag} = transform.structured.generalize %op_{operation_tag} : (!transform.any_op) -> !transform.any_op\n'
        f'    %interchanged_op = transform.structured.interchange %gen_op_{operation_tag} iterator_interchange = {str(interchange_list)} : (!transform.any_op) -> !transform.any_op\n'
        f'    %interchanged_tag = transform.param.constant "{operation_tag}" -> !transform.any_param\n'
        f'    transform.annotate %interchanged_op "tag" = %interchanged_tag : !transform.any_op, !transform.any_param\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}\n'
    )

    __run_transform_code_wrapper(module, transform_code)


def transform_vectorize(module: Module, operation_tag: str):
    """Apply vectorization transformation to an operation.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the operation to transform.
    """
    transform_code = f"""
    module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
            %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.structured.vectorize %op_{operation_tag} : !transform.any_op
            transform.yield
        }}
    }}"""

    __run_transform_code_wrapper(module, transform_code)


def transform_img2col(module: Module, operation_tag: str):
    """Apply img2col transformation to convert convolution to matrix multiplication.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the convolution operation to transform.
    """
    transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op_operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op

    transform.structured.convert_conv2d_to_img2col %op_operation : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }}
}}"""

    __run_transform_code_wrapper(module, transform_code)


def transform_TF(module: Module, consumer_tag: str, producer_tag: str, new_producer_tag: str, tiling_sizes: list[int]):
    """Apply tiling and fusion transformation to consumer and producer operations.

    Tiles the consumer with parallelization and fuses the producer into the tiled loops.

    Args:
        module: The MLIR module to transform.
        consumer_tag: The tag of the consumer operation.
        producer_tag: The tag of the producer operation to fuse.
        new_producer_tag: The tag to assign to the fused producer.
        tiling_sizes: List of tiling factors for consumer loops.
    """
    # If parallel sizes are all zeros, means no fusion will be done
    if all([a == 0 for a in tiling_sizes]):
        return

    transform_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{consumer_tag} = transform.structured.match attributes{{tag = "{consumer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %tiled_op_{consumer_tag}, %forall_op_{consumer_tag} = transform.structured.tile_using_forall %op_{consumer_tag} tile_sizes {str(tiling_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
        f'    %op_{producer_tag} = transform.structured.match attributes{{tag = "{producer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %fused, %containing = transform.structured.fuse_into_containing_op %op_{producer_tag} into %forall_op_{consumer_tag} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
        f'    %fused_tag = transform.param.constant "{new_producer_tag}" -> !transform.any_param\n'
        f'    transform.annotate %fused "tag" = %fused_tag : !transform.any_op, !transform.any_param\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}\n'
    )

    __run_transform_code_wrapper(module, transform_code)


def transform_decompose(module: Module, operation_tag: str):
    """Apply decomposition transformation to an operation.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the operation to decompose.
    """
    transform_code = f"""
    module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %conv = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
            %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op
            %decomposed_tag = transform.param.constant "{operation_tag}" -> !transform.any_param
            transform.annotate %decomposed "tag" = %decomposed_tag : !transform.any_op, !transform.any_param
            transform.yield
        }}
    }}"""

    __run_transform_code_wrapper(module, transform_code)


def transform_transpose_conv_2d(module: Module, operation_tag: str):
    """Apply transposed convolution transformation to an operation.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the convolution operation to transform.
    """
    transform_code = f"""
    module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %conv = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
            %transposed = transform.structured.transpose_conv2d %conv : (!transform.any_op) -> !transform.any_op
            %transposed_tag = transform.param.constant "{operation_tag}" -> !transform.any_param
            transform.annotate %transposed "tag" = %transposed_tag : !transform.any_op, !transform.any_param
            transform.yield
        }}
    }}"""

    __run_transform_code_wrapper(module, transform_code)


def transform_bufferize_and_lower_v(module: Module):
    """Apply bufferization and lowering transformations for vectorized execution.

    Applies a comprehensive series of transformations including bufferization,
    vectorization, and lowering to prepare code for execution.

    Args:
        module: The MLIR module to transform.
    """
    transform_code = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
            %all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.apply_licm to %all_loops : !transform.any_op

            transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
            %empty = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.op<"tensor.empty">
            transform.bufferization.empty_tensor_to_alloc_tensor %empty : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">

            %f0 = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %f0 {
                transform.apply_patterns.vector.transfer_permutation_patterns
                transform.apply_patterns.vector.reduction_to_contract
            } : !transform.any_op
            transform.apply_patterns to %f0 {
                transform.apply_patterns.canonicalization
                transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
            } : !transform.any_op

            %arg1 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op

            %f1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %f1 {
                transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
                transform.apply_patterns.vector.transfer_permutation_patterns
                transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
                transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
                transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
                transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
                transform.apply_patterns.vector.lower_shape_cast
                transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
                transform.apply_patterns.canonicalization
            } : !transform.any_op
            transform.yield
        }
    }"""

    __run_transform_code_wrapper(module, transform_code)


def transform_pre_vec(module: Module, operation_tag: str):
    """Apply pre-vectorization transformation to eliminate unit-stride accesses.

    Eliminates accesses with constant 1 stride by adding subviews, which enables
    better vectorization opportunities.

    Args:
        module: The MLIR module to transform.
        operation_tag: The tag of the operation to transform.
    """
    code_process = subprocess.run(
        f'{os.getenv("PRE_VEC_BIN_PATH")} - {operation_tag}',
        shell=True,
        input=str(module).encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    code = code_process.stdout.decode('utf-8')

    if code_process.returncode != 0:
        raise Exception(code_process.stderr.decode('utf-8'))

    new_module = Module.parse(code, module.context)

    move_module(new_module, module)


def move_module(source: Module, destination: Module):
    """Copy all operations from source module to destination module.

    Args:
        source: The source MLIR module.
        destination: The destination MLIR module where operations will be copied.
    """
    for op in destination.body.operations:
        op.erase()
    for op in source.body.operations:
        destination.body.append(op.clone())


def __run_transform_code_wrapper(module: Module, transform_code: str):
    """Wrapper for running transform code with timeout support.

    Args:
        module: The MLIR module to transform.
        transform_code: The MLIR transform dialect code.
    """
    BindingsProcess.call(__run_transform_code, module, transform_code, timeout=60)


def __run_transform_code(module: Module, transform_code: str):
    """Parse and apply MLIR transform dialect code to a module.

    Args:
        module: The MLIR module to transform.
        transform_code: The MLIR transform dialect code.
    """
    import tempfile
    import sys

    code = str(module)

    code_file = tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False)
    code_file.write(code)
    code_file.close()

    trans_file = tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False)
    trans_file.write(transform_code)
    trans_file.close()

    try:
        script = (
            "import sys\n"
            "from mlir.ir import Context, Module\n"
            "from mlir.dialects.transform import interpreter\n"
            f"with open({code_file.name!r}) as f: code = f.read()\n"
            f"with open({trans_file.name!r}) as f: trans = f.read()\n"
            "with Context():\n"
            "    module = Module.parse(code)\n"
            "    t_module = Module.parse(trans)\n"
            "    interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)\n"
            "    print(str(module))\n"
        )
        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"

        result = subprocess.run(
            [sys.executable, '-c', script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )

        if result.returncode != 0:
            raise Exception(result.stderr or f"Transform failed with exit code {result.returncode}")

        transformed_code = result.stdout

        # Parse and update parent module in-place
        new_module = Module.parse(transformed_code, module.context)
        move_module(new_module, module)

    finally:
        if os.path.exists(code_file.name):
            os.remove(code_file.name)
        if os.path.exists(trans_file.name):
            os.remove(trans_file.name)

