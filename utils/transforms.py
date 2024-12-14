import os
import subprocess
import multiprocessing
import re
from dotenv import load_dotenv
from utils.consts import VECT_TILE_LIMIT
from typing import Any
load_dotenv()


def print_info(*args):
    message = ' '.join(map(str, args))
    print(f"\033[94m[INFO]\t {message}\033[0m")


def print_success(*args):
    message = ' '.join(map(str, args))
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")


def print_alert(*args):
    message = ' '.join(map(str, args))
    print(f"\033[93m[ALERT]\t {message}\033[0m")


def print_error(*args):
    message = ' '.join(map(str, args))
    print(f"\033[91m[ERROR]\t {message}\033[0m")


def get_raw_ast_info(code, tmp_file):

    with open(tmp_file, "w") as file:
        file.write(code)

    result = subprocess.run(
        f'MyASTGenerator/build/bin/AstDumper {tmp_file}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return result.stdout.decode('utf-8')


def get_ast(raw_ast_info):

    info, new_code = raw_ast_info.split("########################################")
    operations_lines, graph_lines = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    ast = {}
    for block in operations_blocks:
        block_lines = block.split('\n')

        operation_tag = block_lines[-2]
        operation = '\n'.join(block_lines[:-3])
        operation = operation.split("#START_NESTED_LOOPS")[0]

        ast[operation_tag] = {
            'producers': [],
            'operation': operation
        }

    graph_lines = graph_lines.split('\n')
    graph_lines = [line.split(' --> ') for line in graph_lines if ' --> ' in line]

    for (producer, consumer) in graph_lines:
        ast[consumer]['producers'].insert(0, producer)

    return ast, new_code.strip()


def extract_loops_data_from_ast_result(raw_ast_info: str, execution_time: float) -> dict[str, Any]:
    info, full_code = raw_ast_info.split("########################################")
    # exec_time = lower_and_run_code(full_code)
    operations_lines, _ = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    operations_details = {}
    operations_details["transform_wrapped_operation"] = full_code
    operations_details["execution_time"] = execution_time
    operations_details['ops_tags'] = []
    for operation_block in operations_blocks:
        loops_detailed = {}
        loops_detailed["nested_loops"] = []
        loops_detailed["op_count"] = {}
        loops_detailed["load_data"] = []
        loops_detailed["store_data"] = []

        operation, rest = operation_block.split("#START_NESTED_LOOPS")
        loops_detailed["operation"] = operation

        nested_loops, rest = rest.split("#START_LOAD_DATA")
        loop_args = []
        for nested_loop in nested_loops.strip().split("\n"):
            if not nested_loop:
                continue
            nested_loop_arr = []
            arg, low, high, step, iter = nested_loop.strip().split(" ")
            nested_loop_arr.extend([f'%{arg}', int(low), int(high), int(step), iter])
            loop_args.append(arg)
            loops_detailed["nested_loops"].append(nested_loop_arr)

        loads_data, rest = rest.split("#START_OP_COUNT")
        for loop_arg in loop_args:
            loads_data = loads_data.replace(loop_arg, f'%{loop_arg}')
        for load_data in loads_data.strip().split("\n"):
            if not load_data:
                continue
            loops_detailed["load_data"].append(load_data.split(", "))

        ops_count, rest = rest.split("#START_TAG")
        for op_count in ops_count.strip().split("\n"):
            op, count = op_count.strip().split(" ")
            loops_detailed["op_count"][op] = int(count)

        operation_tag = rest.strip().split("\n")[0]

        operations_details['ops_tags'].append(operation_tag)
        operations_details[operation_tag] = loops_detailed

    return operations_details

def extract_loops_data_from_code(code: str, execution_time: float):
    result = subprocess.run(
        'MyASTGenerator/build/bin/AstDumper -',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return extract_loops_data_from_ast_result(raw_ast_info, execution_time)

def extract_loops_data_from_file(file_path: str, execution_time: float):
    result = subprocess.run(
        f'MyASTGenerator/build/bin/AstDumper {file_path}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return extract_loops_data_from_ast_result(raw_ast_info, execution_time)


def transform_dialect_TP(code, operation_tag, tiling_size, tmp_file):
    code = code.strip()
    transform_dilaect_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{operation_tag} = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %tiled_op_{operation_tag}, %forall_op_{operation_tag} = transform.structured.tile_using_forall %op_{operation_tag} tile_sizes {str(tiling_size)} : (!transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
        f'    %parallel_op_{operation_tag} = transform.loop.forall_to_parallel %forall_op_{operation_tag} : (!transform.any_op) -> !transform.any_op\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}'
    )

    code = code + transform_dilaect_code

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_tile(code, operation_tag, tiling_size, tmp_file):
    code = code.strip()
    n_loops = sum([s != 0 for s in tiling_size])
    r = ', '.join(['!transform.any_op'] * n_loops)

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

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_interchange(code, operation_tag, interchange_list, tmp_file):
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

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_fuse(code, consumer_tag, producer_tag, tmp_file):
    code = code.strip()

    transform_dilaect_code = (
        f'\nmodule attributes {{transform.with_named_sequence}} {{\n'
        f'  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{\n'
        f'    %op_{producer_tag} = transform.structured.match attributes{{tag = "{producer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %op_{consumer_tag} = transform.structured.match attributes{{tag = "{consumer_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op\n'
        f'    %forall_op_{consumer_tag} = transform.get_parent_op %op_{consumer_tag}: (!transform.any_op) -> !transform.any_op\n'
        f'    transform.structured.fuse_into_containing_op %op_{producer_tag} into %forall_op_{consumer_tag} : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)\n'
        f'    transform.yield\n'
        f'  }}\n'
        f'}}\n'
    )

    code = code + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorise_(code, operation_tag, tmp_file):

    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
    %op_operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op

    transform.structured.vectorize %op_operation : !transform.any_op


    %func = transform.structured.match ops{{["func.func"]}} in %arg1: (!transform.any_op) -> !transform.any_op
    %func_01 = transform.structured.hoist_redundant_vector_transfers %func :(!transform.any_op) -> (!transform.any_op)

    %f = transform.structured.match ops{{["func.func"]}} in %arg1 : (!transform.any_op) -> !transform.any_op

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

    // transform.apply_patterns to %f {{
    //     transform.apply_patterns.vector.reduction_to_contract
    //     transform.apply_patterns.vector.transfer_permutation_patterns
    //     transform.apply_patterns.vector.lower_masked_transfers
    // }} : !transform.any_op
    //
    // transform.apply_patterns to %f {{
    //     transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
    //     transform.apply_patterns.vector.lower_outerproduct
    // }} : !transform.any_op

    transform.yield
  }}
}}""".strip()

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorise_whole_thing(code, tmp_file):

    code = code.strip()

    transform_dilaect_code = """
module attributes {transform.with_named_sequence} {
transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly})
{

   %func = transform.structured.match ops{["func.func"]} in %variant_op
   : (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {vectorize_padding}
    : (!transform.any_op) -> (!transform.any_op)

       // Step 4. Vector backend
  // ======================================================
  %f = transform.structured.match ops{["func.func"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

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
}
""".strip()

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorise_img2col(code, operation_tag, tmp_file):

    code = code.strip()

    transform_dilaect_code = f"""
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

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorise(code, operation_tag, tmp_file):

    code = code.strip()

    transform_dilaect_code = f"""
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

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorise_with_backend(code, operation_tag, tmp_file):

    code = code.strip()

    transform_dilaect_code = f"""
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

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_vectorise_lqcd(code, operation_tag, tmp_file):

    code = code.strip()

    transform_dilaect_code = f"""
module attributes {{transform.with_named_sequence}} {{
    transform.named_sequence @__transform_main(%variant_op: !transform.any_op {{transform.readonly}}) {{
        %operation = transform.structured.match attributes{{tag = "{operation_tag}"}} in %variant_op : (!transform.any_op) -> !transform.any_op
        transform.structured.vectorize %operation : !transform.any_op

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

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def evaluate_code_2(code, tmp_file):
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -buffer-deallocation -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"
    command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    os.environ["OMP_NUM_THREADS"] = "8"

    with open(tmp_file, "w") as file:
        file.write(code)

    os.popen(f"{command_1} {tmp_file} > examples/llvm1.mlir")
    out = os.popen(f"{command_1} {tmp_file} | {command_2} /dev/stdin").read()

    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None


def transform_dialect_img2col(code, operation_tag, tmp_file):

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

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def apply_conv2d_decomposition(code, operation_tag, tmp_file):

    code = code.strip()
    transform_dilaect_code = f"""

        module attributes {{transform.with_named_sequence}} {{
        transform.named_sequence @__transform_main(%arg1: !transform.any_op {{transform.readonly}}) {{
            %conv = transform.structured.match attributes{{tag = "{operation_tag}"}} in %arg1 : (!transform.any_op) -> !transform.any_op
            %decomposed = transform.structured.decompose %conv: (!transform.any_op) -> !transform.any_op
            %decomposed_tag = transform.param.constant "{operation_tag}" -> !transform.any_param
            transform.annotate %decomposed "tag" = %decomposed_tag : !transform.any_op, !transform.any_param


            transform.yield
            }}
        }}"""

    code = code + '\n' + transform_dilaect_code + '\n'

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule",
    ).read()

    result = result.replace("module {\n", "", 1)
    result = ''.join(result.rsplit('\n}\n', 1))
    result = re.sub(r"module attributes \{transform.with_named_sequence\} \{\s+\}", "", result)

    return result


def transform_dialect_prints(code, operation_tags: list, tmp_file):
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

    with open(tmp_file, "w") as file:
        file.write(code)

    result = os.popen(
        f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt {tmp_file} -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule -o {tmp_file}",
    ).read()

    return result


def post_process_transform_dialect_prints(result):

    lines = result.split('\n')
    res = {}

    i = 0
    while i < len(lines):
        if "[[[ IR printer: selected_" in lines[i]:
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


def apply_transformation(state, code, transformation, parameters, from_lqcd):

    tmp_file = state.tmp_file

    code = code.strip()

    # Re-extract loop data if it's gonna be needed afterwards
    if transformation in ['parallelization', 'vectorization']:
        current_dict = extract_loops_data_from_code(code, state.exec_time)
        current_loop_data = current_dict[state.operation_tag]

    if transformation == 'tiling':
        if not parameters:
            print_alert("REASON: No parameters")
            return ''
        new_code = transform_dialect_tile(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'parallelization':
        if not parameters:
            print_alert("REASON: No parameters")
            return ''
        # If a reduction loop is parallelized, ignore the transformation
        for i, (_, _, _, _, iterator_type) in enumerate(current_loop_data['nested_loops']):
            if iterator_type == "reduction" and parameters[i] > 0:
                print_alert("REASON: Reduction parallelization")
                return ''
        new_code = transform_dialect_TP(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'interchange':
        new_code = transform_dialect_interchange(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'img2col':
        new_code = transform_dialect_img2col(code, state.operation_tag, tmp_file)
    elif transformation == 'vectorization':
        # If the operation isn't small enough for vectorization, ignore the transformation
        op_iter_space = 1
        for _, _, upper_bound, _, _ in current_loop_data['nested_loops']:
            op_iter_space *= upper_bound
        if op_iter_space > VECT_TILE_LIMIT:
            print_alert(f"REASON: Too large to vectorize {op_iter_space} > {VECT_TILE_LIMIT}")
            return ''

        if from_lqcd:
            new_code = transform_dialect_vectorise_lqcd(code, state.operation_tag, tmp_file)
        elif state.operation_type == 'conv_2d+img2col':
            new_code = transform_dialect_vectorise_img2col(code, state.operation_tag, tmp_file)
        else:
            new_code = transform_dialect_vectorise(code, state.operation_tag, tmp_file)
    else:
        raise ValueError

    return new_code


def apply_transformation_wrapper(state, code, transformation, parameters, return_list, from_lqcd):
    res = apply_transformation(state, code, transformation, parameters, from_lqcd)
    return_list.append(res)


def apply_transformation_with_timeout(state, code, transformation, parameters, timeout, from_lqcd=False):
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

    return apply_transformation(state, code, transformation, parameters, from_lqcd)


def evaluate_code(code, tmp_file):
    command_1 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize='bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map' -buffer-deallocation -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"
    command_2 = f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs={os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libmlir_c_runner_utils.so,{os.getenv('LLVM_BUILD_PATH')}/lib/libomp.so"

    os.environ["OMP_NUM_THREADS"] = "8"

    with open(tmp_file, "w") as file:
        file.write(code)

    out = os.popen(f"""{command_1} {tmp_file} | {command_2} /dev/stdin""").read()

    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None


def evaluate_code_wrapper(code, return_list, tmp_file):
    res = evaluate_code(code, tmp_file)
    return_list.append(res)


def evaluate_code_with_timeout(code, timeout, tmp_file):
    # manager = multiprocessing.Manager()
    # return_list = manager.list()
    # process = multiprocessing.Process(target=evaluate_code_wrapper, args=(code, return_list, tmp_file))
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

    return evaluate_code(code, tmp_file)
