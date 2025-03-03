import os
import re
import subprocess
from typing import Optional
from utils.log import print_error
from rl_autoschedular import config as cfg
from rl_autoschedular.state import OperationState, OperationFeatures
from rl_autoschedular.observation import get_up_to_date_operation_features
import multiprocessing


# ====================================== Transform dialect functions ======================================

def transform_dialect_TP(code: str, operation_tag: str, tiling_sizes: list[int], tmp_file_path: str):
    """Apply the tiling and parallelization transformation to the specified operation in the given code.

    Args:
        code (str): The code to apply the transformation to.
        operation_tag (str): The tag of the operation to apply the transformation to.
        tiling_size (list[int]): The tiling size to apply.
        tmp_file_path (str): The path to the temporary file to write the code to.

    Returns:
        str: The code after applying the transformation.
    """
    if not code:
        return code

    # If tiling sizes are all zeros, means no tiling is needed
    if all([a == 0 for a in tiling_sizes]):
        return code

    code = code.strip()

    # Add full transform dialect code into the main code
    transform_dialect_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %parallel_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %parallel_tiled_{operation_tag}, %forall_{operation_tag} = transform.structured.tile_using_forall %parallel_{operation_tag} tile_sizes {str(tiling_sizes)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
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
    if not code:
        return code

    # If tiling sizes are all zeros, means no tiling is needed
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
    if not code:
        return code

    # If the permutation list is same as the identity permutation, means no interchange is needed
    if interchange_list == list(range(len(interchange_list))):
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
    if not code:
        return code

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


def transform_dialect_vectorise(code: str, operation_tag: str, tmp_file_path: str):
    """Apply the vectorization transformation to the specified operation in the given code.

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
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}})
{{

  // %conv_gen_2 = transform.structured.match attributes{{tag = "{operation_tag}"}} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %forall_op = transform.get_parent_op %conv_gen_2: (!transform.any_op) -> !transform.any_op

  %forall_op = transform.structured.match ops{{["scf.forall"]}}  in %variant_op : (!transform.any_op) -> !transform.any_op


  %original_fill = transform.structured.match ops{{["linalg.fill"]}} in %variant_op : (!transform.any_op) -> !transform.any_op
  transform.structured.fuse_into_containing_op %original_fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  %func = transform.structured.match ops{{["func.func"]}} in %variant_op: (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {{vectorize_padding}}: (!transform.any_op) -> (!transform.any_op)

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


def transform_dialect_vectorise_with_vectorizer(code: str, operation_tag: str, tmp_file_path: str):
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

    vect_code_process = subprocess.run(
        f'{os.getenv("VECTORIZER_BIN_PATH")} - {operation_tag}',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    vect_code = vect_code_process.stdout.decode('utf-8')

    if vect_code_process.returncode != 0:
        print_error(f"Vectorizer failed with error: {vect_code_process.stderr.decode('utf-8')}")
        return ''

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
    if not code:
        return code

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


def apply_transformation(state: OperationState, code: str, transformation: str, parameters: list, use_vectorizer: bool = cfg.use_vectorizer) -> str:
    """Apply the specified transformation to the given code.

    Args:
        state (OperationState): The operation state.
        bench_features (BenchmarkFeatures): The benchmark features.
        code (str): The code to apply the transformation to.
        transformation (str): The transformation to apply.
        parameters (list): The parameters of the transformation.
        use_vectorizer (bool): Whether to use the vectorizer or not.

    Returns:
        str: The code after applying the transformation.
    """
    # Operation features are needed for parallelization and vectorization
    if transformation in ['parallelization', 'vectorization']:
        operation_features = get_up_to_date_operation_features(state)

    tmp_file = state.tmp_file

    code = code.strip()

    if transformation == 'tiling':
        new_code = transform_dialect_tile(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'parallelization':
        parallel_params = [0 if operation_features.nested_loops[i].iterator_type == "reduction" else parameters[i] for i in range(len(parameters))]
        tiling_params = [parameters[i] if operation_features.nested_loops[i].iterator_type == "reduction" else 0 for i in range(len(parameters))]
        new_code = transform_dialect_TP(code, state.operation_tag, parallel_params, tmp_file)
        new_code = transform_dialect_tile(new_code, state.operation_tag, tiling_params, tmp_file)
    elif transformation == 'interchange':
        new_code = transform_dialect_interchange(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'img2col':
        new_code = transform_dialect_img2col(code, state.operation_tag, tmp_file)
    elif transformation == 'vectorization':
        assert is_vectorizable(operation_features), "Too large to vectorize"

        if use_vectorizer:
            new_code = transform_dialect_vectorise_with_vectorizer(code, state.operation_tag, tmp_file)
        elif state.operation_type == 'conv_2d' and (i := state.last_op_history_index()) is not None and state.transformation_history[i][0] == 'img2col':
            new_code = transform_dialect_vectorise_img2col(code, state.operation_tag, tmp_file)
        else:
            new_code = transform_dialect_vectorise(code, state.operation_tag, tmp_file)
    elif transformation == 'no_transformation':
        new_code = code
    else:
        raise ValueError(f"Invalid transformation: {transformation}")

    return new_code


def apply_transformation_wrapper(state: OperationState, code: str, transformation: str, parameters: list, use_vectorizer: bool, return_list):
    """Wrapper function to apply the transformation with multiprocessing.

    Args:
        state (OperationState): The operation state.
        bench_features (BenchmarkFeatures): The benchmark features.
        code (str): The code to apply the transformation to.
        transformation (str): The transformation to apply.
        parameters (list): The parameters of the transformation.
        return_list (list): The list to store the result of the transformation.
        use_vectorizer (bool): Whether to use the vectorizer or not. Default is False.
    """
    res = apply_transformation(state, code, transformation, parameters, use_vectorizer)
    return_list.append(res)


def apply_transformation_with_timeout(state: OperationState, code: str, transformation: str, parameters: list, use_vectorizer: bool = cfg.use_vectorizer, timeout: Optional[float] = 20) -> str:
    """Apply the specified transformation to the given code with a timeout.

    Args:
        state (OperationState): The operation state.
        bench_features (BenchmarkFeatures): The benchmark features.
        code (str): The code to apply the transformation to.
        transformation (str): The transformation to apply.
        parameters (list): The parameters of the transformation.
        timeout (int): The timeout in seconds.
        use_vectorizer (bool): Whether to use the vectorizer or not.

    Returns:
        str: The code after applying the transformation.
    """
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=apply_transformation_wrapper, args=(state, code, transformation, parameters, use_vectorizer, return_list))
    process.start()
    process.join(timeout)

    if process.is_alive():
        # The function is still running, terminate the process
        process.terminate()
        process.join()

        return None
    else:
        # The function completed within the timeout
        return return_list[0]


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


def is_vectorizable(operation_features: OperationFeatures) -> bool:
    """Check if the operation is small enough for vectorization.

    Args:
        operation_features (OperationFeatures): The operation features.

    Returns:
        bool: Whether the operation is vectorizable or not.
    """
    op_iter_space = 1
    for nested_loop in operation_features.nested_loops:
        op_iter_space *= nested_loop.upper_bound
    return op_iter_space <= cfg.vect_size_limit
