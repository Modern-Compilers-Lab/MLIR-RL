import os
import subprocess
from mlir.ir import Context, Module
from mlir.dialects.transform import interpreter
from utils.bindings_process import BindingsProcess


def transform_TP(code: str, operation_tag: str, tiling_sizes: list[int]):
    """Apply the tiling and parallelization transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tiling_sizes (list[int]): The tiling size to apply.

    Returns:
        str: The code after applying the transformation.
    """
    # If tiling sizes are all zeros, means no tiling is needed
    if all([a == 0 for a in tiling_sizes]):
        return code

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

    return __run_transform_code(code, transform_code)


def transform_tile(code: str, operation_tag: str, tiling_sizes: list[int]):
    """Apply the tiling transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tiling_sizes (list[int]): The tiling size to apply.

    Returns:
        str: The code after applying the transformation.
    """
    # If tiling sizes are all zeros, means no tiling is needed
    if all([a == 0 for a in tiling_sizes]):
        return code

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

    return __run_transform_code(code, transform_code)


def transform_interchange(code: str, operation_tag: str, interchange_list: list[int]):
    """Apply the interchange transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        interchange_list (list[int]): The interchange list to apply.

    Returns:
        str: The code after applying the transformation.
    """
    # If the permutation list is same as the identity permutation, means no interchange is needed
    if interchange_list == list(range(len(interchange_list))):
        return code

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

    return __run_transform_code(code, transform_code)


def transform_vectorize_img2col(code: str, operation_tag: str):
    """Apply the vectorization transformation with img2col to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
    """
    transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}})
{{

  // %conv_gen_2 = transform.structured.match attributes{{tag = "{operation_tag}"}} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %forall_op = transform.get_parent_op %conv_gen_2: (!transform.any_op) -> !transform.any_op

  %forall_op = transform.structured.match ops{{["scf.forall"]}}  in %variant_op : (!transform.any_op) -> !transform.any_op



  %producer = transform.structured.match attributes{{tag = "img2col_producer"}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %producer into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %fb = transform.structured.match ops{{["func.func"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %fb {{
    transform.apply_patterns.canonicalization
  }} : !transform.any_op
  transform.apply_cse to %fb : !transform.any_op


  %original_fill = transform.structured.match ops{{["linalg.fill"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %original_fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %fb1 = transform.structured.match ops{{["func.func"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %fb1 {{
    transform.apply_patterns.canonicalization
  }} : !transform.any_op
  transform.apply_cse to %fb1 : !transform.any_op



   %func = transform.structured.match ops{{["func.func"]}} in %variant_op
   : (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {{vectorize_padding}}
    : (!transform.any_op) -> (!transform.any_op)

       // Step 4. Vector backend
  // ======================================================
  %f = transform.structured.match ops{{["func.func"]}} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %f {{
    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    transform.apply_patterns.vector.transfer_permutation_patterns
    transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
    transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
    transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
    transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
    transform.apply_patterns.vector.lower_shape_cast
    transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
    transform.apply_patterns.canonicalization
  }} : !transform.any_op



  transform.yield
}}
}}
"""

    return __run_transform_code(code, transform_code)


def transform_vectorize_children(code: str):
    """Apply the vectorization transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
    """
    transform_code = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly})
        {
            %forall_op = transform.structured.match ops{["scf.forall"]}  in %variant_op : (!transform.any_op) -> !transform.any_op

            %original_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
            transform.structured.fuse_into_containing_op %original_fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

            %func = transform.structured.match ops{["func.func"]} in %variant_op: (!transform.any_op) -> !transform.any_op
            %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {vectorize_padding}: (!transform.any_op) -> (!transform.any_op)

            transform.yield
        }
    }"""

    return __run_transform_code(code, transform_code)


def transform_vectorize_with_vectorizer(code: str, operation_tag: str):
    """Apply the vectorization transformation with vectorizer to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
    """
    vect_code_process = subprocess.run(
        f'{os.getenv("VECTORIZER_BIN_PATH")} - {operation_tag}',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    vect_code = vect_code_process.stdout.decode('utf-8')

    if vect_code_process.returncode != 0:
        raise Exception(vect_code_process.stderr.decode('utf-8'))

    return vect_code


def transform_vectorize(code: str, operation_tag: str):
    """Apply the vectorization transformation with vectorizer to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
    """
    transform_code = f"""
    module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg0: !transform.any_op {{transform.readonly}}) {{
            %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg0 : (!transform.any_op) -> !transform.any_op
            transform.structured.vectorize %op_{operation_tag} : !transform.any_op
            transform.yield
        }}
    }}"""

    return __run_transform_code(code, transform_code)


def transform_img2col(code: str, operation_tag: str):
    """Apply the img2col transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
    """
    transform_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op_operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op

    transform.structured.convert_conv2d_to_img2col %op_operation : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    transform.yield
  }}
}}"""
    # // %a_tag = transform.param.constant "img2col_producer" -> !transform.any_param
    # // transform.annotate %a "tag" = %a_tag : !transform.any_op, !transform.any_param

    # // %matmul_op = transform.get_producer_of_operand %b[0]: (!transform.any_op) -> !transform.any_op
    # // %matmul_op_tag = transform.param.constant "{operation_tag}" -> !transform.any_param
    # // transform.annotate %matmul_op "tag" = %matmul_op_tag : !transform.any_op, !transform.any_param

    return __run_transform_code(code, transform_code)


def transform_TF(code: str, consumer_tag: str, producer_tag: str, new_producer_tag: str, tiling_sizes: list[int]):
    """Apply the tiling and fusion transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        consumer_tag (str): The tag of the operation to apply the transformation to.
        producer_tag (str): the tag of the producer to fuse with
        new_producer_tag (str): the tag to assign to the producer after fusion.
        tiling_sizes (list[int]): The tiling size to apply.
        parallel_sizes (list[int]): The parallel size to apply.

    Returns:
        str: The code after applying the transformation.
    """
    # If parallel sizes are all zeros, means no fusion will be done
    if all([a == 0 for a in tiling_sizes]):
        return code

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

    return __run_transform_code(code, transform_code)


def transform_decompose(code: str, operation_tag: str):
    """Apply the decomposition transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
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

    return __run_transform_code(code, transform_code)


def transform_transpose_conv_2d(code: str, operation_tag: str):
    """Apply the Conv2D transpose transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
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

    return __run_transform_code(code, transform_code)


def transform_bufferize_and_lower_v(code: str):
    """Apply the vectorization transformation with vectorizer to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
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

    return __run_transform_code(code, transform_code)


def transform_pre_vec(code: str, operation_tag: str):
    """Eliminate accesses with the constant 1 by adding subviews
    which enables more vectorization.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.

    Returns:
        str: The code after applying the transformation.
    """
    code_process = subprocess.run(
        f'{os.getenv("PRE_VEC_BIN_PATH")} - {operation_tag}',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    code = code_process.stdout.decode('utf-8')

    if code_process.returncode != 0:
        raise Exception(code_process.stderr.decode('utf-8'))

    return code


def __run_transform_code(code: str, transform_code: str):
    def transform_bind_call():
        with Context():
            module = Module.parse(code)
            t_module = Module.parse(transform_code)
        interpreter.apply_named_sequence(module, t_module.body.operations[0], t_module)

        return str(module)

    return BindingsProcess.call(transform_bind_call, timeout=60)
