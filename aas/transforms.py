from aas import config as cfg
from aas.state import OperationState
from aas.observation.operation import NestedLoopFeatures, extract_op_features_from_code
from aas.action import Action, Parallelization, Vectorization, NoTransformation
from utils.log import print_alert
import os
import re
import subprocess
from typing import Optional


# ====================================== Transform dialect functions ======================================

def transform_dialect_TP(code: str, operation_tag: str, tiling_size: list[Optional[int]], nested_loops_features: list[NestedLoopFeatures], tmp_file_path: str):
    """Apply the tiling and parallelization transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tiling_size (list[Optional[int]]): The tiling size to apply.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """
    if not tiling_size or None in tiling_size:
        return ''
    # Crop tiling sizes to the number of loops
    tiling_size = tiling_size[:len(nested_loops_features)]
    # If all tiling sizes are 0, return the code as is
    if all([a == 0 for a in tiling_size]):
        return code

    code = code.strip()

    # Set tiling with parallelization for loops that can be parallelized
    parallel_tiling_sizes = [0 if nested_loops_features[i].iterator_type == "reduction" else tiling_size[i] for i in range(len(tiling_size))]
    if any([a != 0 for a in parallel_tiling_sizes]):
        parallel_transform_dialect_code = (
            f'    %parallel_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
            f'    %parallel_tiled_{operation_tag}, %forall_{operation_tag} = transform.structured.tile_using_forall %parallel_{operation_tag} tile_sizes {str(parallel_tiling_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
        )
    else:
        parallel_transform_dialect_code = ''

    reduction_tiling_sizes = [tiling_size[i] if nested_loops_features[i].iterator_type == "reduction" else 0 for i in range(len(tiling_size))]
    reduction_loop_sizes = [nested_loops_features[i].upper_bound - nested_loops_features[i].lower_bound if nested_loops_features[i].iterator_type == "reduction" else 0 for i in range(len(nested_loops_features))]
    if cfg.parallelize_reduction:
        # Set tiling with parallelization for reduction loops
        if any([a != 0 for a in reduction_tiling_sizes]):
            reduction_transform_dialect_code = (
                f'    %reduction_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f'    %reduction_fill_{operation_tag}, %reduction_tiled_{operation_tag}, %reduction_comb_{operation_tag}, %reduction_forall_{operation_tag} = transform.structured.tile_reduction_using_forall %reduction_{operation_tag} by num_threads = {str([loop_size // tile_size if loop_size > 0 else 0 for loop_size, tile_size in zip(reduction_loop_sizes, reduction_tiling_sizes)])}, tile_sizes = [] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)\n'
            )
        else:
            reduction_transform_dialect_code = ''
    else:
        # Set tiling only for reduction loops
        nb_reduction_loops = sum([s != 0 for s in reduction_tiling_sizes])
        if nb_reduction_loops > 0:
            reduction_transform_dialect_code = (
                f'    %reduction_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
                f'    %reduction_tiled_{operation_tag}, %loops_{operation_tag}{f':{nb_reduction_loops}' if nb_reduction_loops > 1 else ''} = transform.structured.tile_using_for %reduction_{operation_tag} tile_sizes {str(reduction_tiling_sizes)} : (!transform.any_op) -> (!transform.any_op{', !transform.any_op' * nb_reduction_loops})\n'
            )
        else:
            reduction_transform_dialect_code = ''

    # Add full transform dialect code into the main code
    transform_dialect_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'{parallel_transform_dialect_code}'
        f'{reduction_transform_dialect_code}'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}'
    )
    code = code + transform_dialect_code

    with open(tmp_file_path, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_tile(code: str, operation_tag: str, tiling_size: list[int], tmp_file_path: str):
    """Apply the tiling transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tiling_size (list[int]): The tiling size to apply.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """
    if not tiling_size:
        return ''
    if all([a == 0 for a in tiling_size]):
        return code

    code = code.strip()
    n_loops = sum([s != 0 for s in tiling_size])
    r = ', '.join(['!transform.any_op'] * n_loops)
    assert n_loops > 0, "No loops to tile"

    transform_dilaect_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %tiled_op_{operation_tag}, %loops:{n_loops} = transform.structured.tile_using_for %op_{operation_tag} tile_sizes {str(tiling_size)} : (!transform.any_op) -> (!transform.any_op, {r})\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}\n'
    )

    code = code + transform_dilaect_code + '\n'

    with open(tmp_file_path, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_interchange(code: str, operation_tag: str, interchange_list: list[int], tmp_file_path: str):
    """Apply the interchange transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        interchange_list (list[int]): The interchange list to apply.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """
    if not interchange_list:
        return code

    code = code.strip()

    transform_dilaect_code = (
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

    code = code + transform_dilaect_code + '\n'

    with open(tmp_file_path, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


# def transform_dialect_fuse(code, consumer_tag, producer_tag, tmp_file):
#     code = code.strip()

#     transform_dilaect_code = (
#         f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
#         f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
#         f'    %op_{producer_tag} = transform.structured.match attributes{{tag = "{producer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
#         f'    %op_{consumer_tag} = transform.structured.match attributes{{tag = "{consumer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
#         f'    %forall_op_{consumer_tag} = transform.get_parent_op %op_{consumer_tag}: (!transform.any_op) -> !transform.any_op\n'
#         f'    transform.structured.fuse_into_containing_op %op_{producer_tag} into %forall_op_{consumer_tag} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
#         f'    transform.yield\n'
#         f'  }}\n'
#         f'}}\n'
#     )

#     code = code + transform_dilaect_code + '\n'

#     with open(tmp_file, "w") as file:
#         file.write(code)

#     result = os.popen(
#         f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
#     ).read()

#     result = result.replace("module {\n", "", 1)
#     result = ''.join(result.rsplit('\n}\n', 1))
#     result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

#     return result


def transform_dialect_vectorise_img2col(code: str, operation_tag: str, tmp_file_path: str):
    """Apply the vectorization transformation with img2col to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """

    code = code.strip()

    transform_dialect_code = f"""
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
""".strip()

    code = code + '\n' + transform_dialect_code + '\n'

    with open(tmp_file_path, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorize(code: str, operation_tag: str, tmp_file_path: str):
    """Apply the vectorization transformation with vectorizer to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """
    if not code:
        return code

    code = code.strip()

    transform_dialect_code = f"""
    module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}}) {{
            %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %variant_op : (!transform.any_op) -> !transform.any_op
            transform.structured.vectorize %op_{operation_tag} : !transform.any_op

            %f = transform.structured.match ops{{[\"func.func\"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %f {{
                transform.apply_patterns.vector.transfer_permutation_patterns
                transform.apply_patterns.vector.reduction_to_contract
                transform.apply_patterns.canonicalization
                transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
            }} : !transform.any_op

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
    }}""".strip()

    full_code = code + '\n' + transform_dialect_code + '\n'

    with open(tmp_file_path, "w") as file:
        file.write(full_code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorise_with_vectorizer(code: str, operation_tag: str, tmp_file_path: str):
    """Apply the vectorization transformation with vectorizer to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """

    code = code.strip()

    vect_code_process = subprocess.run(
        f'{os.getenv("VECTORIZER_BIN_PATH")} - {operation_tag}',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    vect_code = vect_code_process.stdout.decode('utf-8')

    # If vectorizer succeeded apply vectorization patterns else return empty string
    if vect_code:
        transform_dialect_code = """
    module attributes {transform.with_named_sequence} {
        transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) {
            %f = transform.structured.match ops{[\"func.func\"]} in %variant_op : (!transform.any_op) -> !transform.any_op
            transform.apply_patterns to %f {
                transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
                transform.apply_patterns.vector.transfer_permutation_patterns
                transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
                transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
                transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
                transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
                transform.apply_patterns.vector.lower_shape_cast
                transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
                transform.apply_patterns.canonicalization
            } : !transform.any_op
            transform.yield
        }
    }""".strip()

        full_code = vect_code + '\n' + transform_dialect_code + '\n'

        with open(tmp_file_path, "w") as file:
            file.write(full_code)

        result = os.popen(
            f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
        ).read()

        result = result.replace("module {\n", "", 1)
        result = ''.join(result.rsplit('\n}\n', 1))
        result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

        return result
    else:
        return ''


def transform_dialect_img2col(code: str, operation_tag: str, tmp_file_path: str):
    """Apply the img2col transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """

    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op_operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op

    %a, %b = transform.structured.convert_conv2d_to_img2col %op_operation : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    %a_tag = transform.param.constant "img2col_producer" -> !transform.any_param
    transform.annotate %a "tag" = %a_tag : !transform.any_op, !transform.any_param

    %matmul_op = transform.get_producer_of_operand %b[0]: (!transform.any_op) -> !transform.any_op
    %matmul_op_tag = transform.param.constant "{operation_tag}" -> !transform.any_param
    transform.annotate %matmul_op "tag" = %matmul_op_tag : !transform.any_op, !transform.any_param

    transform.yield
  }}
}}""".strip()

    code = code + transform_dilaect_code

    with open(tmp_file_path, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def apply_transformation(state: OperationState, code: str, tmp_file_path: str, action: Action, use_vectorizer: bool = False):
    """Apply the specified transformation to the given code.

    Args:
        state (OperationState): The operation state.
        code (str): The code to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to.
        action (Action): The transformation to apply.
        use_vectorizer (bool): Whether to use the vectorizer or not.

    Returns:
        str: The code after applying the transformation.
    """

    code = code.strip()

    # Re-extract loop data if it's gonna be needed afterwards
    if isinstance(action, Parallelization) or isinstance(action, Vectorization):
        operation_features = extract_op_features_from_code(code, state.operation_tag)

    # if action.name == 'tiling':
    #     if not action.params:
    #         print_alert("REASON: No parameters")
    #         return ''
    #     new_code = transform_dialect_tile(code, state.operation_tag, action.params, tmp_file_path)
    if isinstance(action, Parallelization):
        if not action.params:
            print_alert("REASON: No parameters")
            return ''
        new_code = transform_dialect_TP(code, state.operation_tag, action.params, state.operation_features.nested_loops, tmp_file_path)
    # elif transformation == 'interchange':
    #     new_code = transform_dialect_interchange(code, state.operation_tag, parameters, tmp_file_path)
    # elif transformation == 'img2col':
    #     new_code = transform_dialect_img2col(code, state.operation_tag, tmp_file_path)
    elif isinstance(action, Vectorization):
        # Force no transformation on pooling operations
        if state.operation_features.operation_type == 'pooling':
            return code

        # If the operation isn't small enough for vectorization, ignore the transformation
        if not Vectorization.is_possible(operation_features):
            if operation_features.op_iter_space_size > cfg.vect_size_limit:
                print_alert(f"REASON: Too large to vectorize {operation_features.op_iter_space_size} > {cfg.vect_size_limit}")
            else:
                print_alert("REASON: Operation is not vectorizable")
            return ''

        # # For convolution, before vectorization, we need to first apply another tiling in order to decompose it to 1d convolution
        # if (state.operation_features.operation_type == 'conv_2d'):
        #     if ('conv_2d_nhwc_hwcf' in state.operation_features.raw_operation):
        #         second_interchange_parameters = parameters.copy()
        #         second_interchange_parameters[1] = 1
        #         second_interchange_parameters[4] = 1
        #     elif ('conv_2d_nchw_fchw' in state.operation_features.raw_operation):
        #         second_interchange_parameters = parameters.copy()
        #         second_interchange_parameters[2] = 1
        #         second_interchange_parameters[5] = 1
        #     elif ('pooling' in state.operation_features.raw_operation):
        #         second_interchange_parameters = [0] * 6
        #         second_interchange_parameters[2] = 1
        #         second_interchange_parameters[4] = 1
        #     state.transformed_code = apply_transformation_with_timeout(
        #         state=state,
        #         bench_features=bench_data,
        #         code=state.transformed_code,
        #         transformation='tiling',
        #         parameters=second_interchange_parameters,
        #         timeout=20,
        #         use_vectorizer=cfg.use_vectorizer
        #     )

        #     state.transformed_code = apply_conv2d_decomposition(state.transformed_code, state.operation_tag, self.tmp_file)

        if use_vectorizer:
            new_code = transform_dialect_vectorise_with_vectorizer(code, state.operation_tag, tmp_file_path)
        elif state.operation_features.operation_type == 'conv_2d+img2col':
            new_code = transform_dialect_vectorise_img2col(code, state.operation_tag, tmp_file_path)
        else:
            new_code = transform_dialect_vectorize(code, state.operation_tag, tmp_file_path)
    elif isinstance(action, NoTransformation):
        new_code = code
    else:
        raise ValueError(f"Invalid transformation: {action}")

    return new_code


def apply_transformation_wrapper(state: OperationState, code: str, tmp_file_path: str, action: Action, return_list, use_vectorizer: bool = False):
    """Wrapper function to apply the transformation with multiprocessing.

    Args:
        state (OperationState): The operation state.
        code (str): The code to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to.
        action (Action): The transformation to apply.
        return_list (list): The list to store the result of the transformation.
        use_vectorizer (bool): Whether to use the vectorizer or not. Default is False.
    """
    res = apply_transformation(state, code, tmp_file_path, action, use_vectorizer)
    return_list.append(res)


def apply_transformation_with_timeout(state: OperationState, code: str, tmp_file_path: str, action: Action, timeout: Optional[float] = None, use_vectorizer: bool = False):
    """Apply the specified transformation to the given code with a timeout.

    Args:
        state (OperationState): The operation state.
        code (str): The code to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to
        action (Action): The transformation to apply.
        timeout (int): The timeout in seconds.
        use_vectorizer (bool): Whether to use the vectorizer or not.

    Returns:
        str: The code after applying the transformation.
    """
    # manager = multiprocessing.Manager()
    # return_list = manager.list()
    # process = multiprocessing.Process(target=apply_transformation_wrapper, args=(state, code, transformation, parameters, return_list, from_lqcd))
    # process.start()
    # process.join(timeout)

    # if process.is_alive():
    #     # The function is still running, terminate the process
    #     process.terminate()
    #     process.join()

    #     return None
    # else:
    #     # The function completed within the timeout
    #     return return_list[0]

    return apply_transformation(state, code, tmp_file_path, action, use_vectorizer)


# ========================================= Other functions =========================================


def apply_conv2d_decomposition(code: str, operation_tag: str, tmp_file_path: str):
    """Apply the Conv2D decomposition transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """
    code = code.strip()
    transform_dialect_code = f"""
        module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %conv = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
            %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op
            %decomposed_tag = transform.param.constant "{operation_tag}" -> !transform.any_param
            transform.annotate %decomposed "tag" = %decomposed_tag : !transform.any_op, !transform.any_param
            transform.yield
            }}
        }}"""

    code = code + '\n' + transform_dialect_code + '\n'

    with open(tmp_file_path, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def get_ops_by_tags(code: str, operation_tags: list, tmp_file_path: str):
    """Get operations by using tags in the specified code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tags (list): The list of tags of the operations to print.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        dict[str, str]: containing for each opeartion tag the corresponding operation.
    """
    matchs = '\n'.join([f""" %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op """ for operation_tag in operation_tags])
    prints = '\n'.join([f""" transform.print %op_{operation_tag} {{name = "selected_{operation_tag}"}}: !transform.any_op """ for operation_tag in operation_tags])

    code = code.strip()
    transform_dilaect_code = f"""
        module attributes {{transform.with_named_sequence}} {{
            transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
                {matchs}

                {prints}

                transform.yield
            }}
        }}"""

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file_path, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file_path} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule -o {tmp_file_path}",
    ).read()

    lines = result.split('\n')
    res = {}

    i = 0
    while i < len(lines):
        if "[[[ IR printer: selected_" in lines[i]:
            # TODO: Find out another way to do this (current solution may introduce bugs)
            opreation_id = lines[i][25:-4]

            operation = []
            i += 1
            while i < len(lines) and not (("[[[ IR printer: selected_" in lines[i]) or (" = affine_map<" in lines[i]) or ("module attributes" in lines[i])):
                operation.append(lines[i])
                i += 1

            operation = '\n'.join(operation)
            operation = ' '.join(operation.split(' ')[2:])
            res[opreation_id] = operation

        else:
            i += 1

    return res
