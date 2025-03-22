import numpy as np
import re
from typing import Optional
import os
from copy import copy
import subprocess
from rl_autoschedular import config as cfg
from rl_autoschedular.state import OperationFeatures, NestedLoopFeatures, BenchmarkFeatures, OperationState, OperationType


# ================================================ Public functions ================================================


def build_op_features_vector(op_features: OperationFeatures):
    """Build the feature vector from the operation features dataclass.

    Args:
        op_features (OperationFeatures): the operation features

    Returns:
        np.ndarray: the feature vector
    """

    indices = [nested_loop.arg for nested_loop in op_features.nested_loops]
    indices_dim = {arg: i for (i, arg) in enumerate(indices)}

    # Nested loop features: (upper/lower bounds, step)
    nested_loops = np.zeros((cfg.max_num_loops,))
    for i, nested_loop in enumerate(op_features.nested_loops):
        if i == cfg.max_num_loops:
            break
        nested_loops[i] = nested_loop.upper_bound

    # load access matrices:
    load_data = op_features.load_data

    load_access_matrices = np.zeros((cfg.max_num_stores_loads, cfg.max_num_load_store_dim, cfg.max_num_loops), dtype=np.int16)

    for load_i, load in enumerate(load_data):
        if load_i == cfg.max_num_stores_loads:
            break
        dimensions_terms = [__formula_str_to_list(term) for term in load]
        for m, dimension_term in enumerate(dimensions_terms):
            for index, factor in dimension_term:
                if index in indices_dim:
                    n = indices_dim[index]
                    load_access_matrices[load_i, m, n] = factor

    # store access matrices:
    store_data = op_features.store_data

    store_access_matrices = np.zeros((cfg.max_num_load_store_dim, cfg.max_num_loops), dtype=np.int16)

    dimensions_terms = [__formula_str_to_list(term) for term in store_data]
    for m, dimension_term in enumerate(dimensions_terms):
        for index, factor in dimension_term:
            n = indices_dim[index]
            store_access_matrices[m, n] = factor

    # Operations count:
    operations_count = np.array(list(op_features.op_count.values()))

    # Feature vector:
    nested_loops = nested_loops.reshape(-1)
    load_access_matrices = load_access_matrices.reshape(-1)
    store_access_matrices = store_access_matrices.reshape(-1)

    feature_vector = np.concatenate((nested_loops, load_access_matrices, store_access_matrices, operations_count))

    return feature_vector


def extract_op_features_from_affine_code(raw_operation: str, tmp_file_path: str):
    """Get operation features from the raw operation.

    Args:
        raw_operation (str): the raw operation
        tmp_file_path (str): the temporary file path to write the operation to

    Returns:
        OperationFeatures: operation features contained in the raw operation
    """
    # Get code as affine loops
    operation_type = __get_operation_type(raw_operation)
    wrapped_operation = __function_wrapper(raw_operation)
    loops = __lower_linalg_to_loops(wrapped_operation, tmp_file_path)
    lines = loops.split('\n') if loops else []

    # Build op features
    nested_loops = []
    op_count = {'+': 0, '-': 0, '*': 0, '/': 0, 'exp': 0}
    load_data = []
    store_data = []

    maps: dict[str, str] = {}
    args_of_loops: list[str] = []
    args_of_map: dict[str, str] = {}

    for line in lines:

        if "affine_map" in line:
            map_name, map_function = line.strip().split(' = ')
            map_function = map_function.split(' -> ')[1][1:-2]
            maps[map_name] = map_function

        elif "affine.apply" in line:
            new_op, _, _, *map_name__args = line.strip().split(' ')
            map_name__args = ' '.join(map_name__args)
            s = map_name__args.index('(')
            map_name, args = map_name__args[:s], map_name__args[s + 1:-1].split(', ')
            mapping_string = copy(maps[map_name])
            for i in range(len(args)):
                mapping_string = mapping_string.replace(f'd{i}', args[i])
            args_of_map[new_op] = mapping_string

        elif "affine.for" in line:
            _, arg, _, lower, _, upper, _ = line.strip().split(' ')
            # TODO: handle iterator types better
            nested_loops.append(
                NestedLoopFeatures(
                    arg=arg,
                    lower_bound=int(lower),
                    upper_bound=int(upper),
                    step=1,
                    iterator_type='parallel'
                )
            )
            args_of_loops.append(arg)

        elif "affine.load" in line:
            new_op, _, _, *alloc = line.strip().split(' ')[:-2]
            alloc = ' '.join(alloc)
            args = alloc.split('[')[1][:-1].split(', ')

            for i in range(len(args)):
                if args[i] in args_of_map:
                    args[i] = args_of_map[args[i]]

            load_data.append(args)

        elif "arith.addf" in line:
            op_count['+'] += 1
        elif "arith.mulf" in line:
            op_count['*'] += 1
        elif "arith.subf" in line:
            op_count['-'] += 1
        elif "arith.divf" in line:
            op_count['/'] += 1
        elif "math.exp" in line:
            op_count['exp'] += 1

    return OperationFeatures(
        raw_operation=raw_operation,
        operation_type=operation_type,
        op_count=op_count,
        load_data=load_data,
        store_data=store_data,
        nested_loops=nested_loops,
        vectorizable=True
    )


def extract_bench_features_from_code(bench_name: str, code: str, root_execution_time: int):
    """Extract benchmark features from the given code.

    Args:
        bench_name (str): the benchmark name
        code (str): the code to extract features from
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

    Returns:
        BenchmarkFeatures: the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} -',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time)


def extract_bench_features_from_file(bench_name: str, file_path: str, root_execution_time: int):
    """Extract benchmark features from the code in the file.

    Args:
        bench_name (str): the benchmark name
        file_path (str): the file path
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

    Returns:
        BenchmarkFeatures: the extracted benchmark features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} {file_path}',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_bench_features_from_ast_result(bench_name, raw_ast_info, root_execution_time)


def update_operation_features(state: OperationState, transformation: str, parameters: list[int]) -> OperationFeatures:
    """Update the operation features after applying a transformation.

    Args:
        operation_features (OperationFeatures): The operation features.
        transformation (str): The transformation name.
        parameters (list[int]): The transformation parameters.

    Returns:
        OperationFeatures: The updated operation features.
    """
    new_operation_features = state.operation_features.copy()
    if transformation in ['no_transformation', 'vectorization']:
        return new_operation_features

    match transformation:
        case 'parallelization' | 'tiling':
            for nested_loop, tile_size in zip(new_operation_features.nested_loops, parameters):
                if tile_size == 0:
                    continue
                nested_loop.upper_bound = tile_size
        case 'interchange':
            for i, j in enumerate(parameters):
                new_operation_features.nested_loops[i] = state.operation_features.nested_loops[j]
        case _:
            raise ValueError(f"Invalid transformation: {transformation}")

    return new_operation_features


# def update_operation_features_from_scratch(state: OperationState) -> OperationFeatures:
#     """Update the operation features iteratively from scratch.

#     Notes:
#         Should only be used when operation features haven't been updated before.
#         i.e: cfg.update_op_features = False

#     Args:
#         state (OperationState): The current state.

#     Returns:
#         OperationFeatures: The updated operation features.
#     """
#     assert not cfg.update_op_features

#     state_copy = state.copy()

#     # Get the path to traverse
#     path_start = state_copy.last_op_history_index()
#     if path_start is None:
#         return state_copy.operation_features
#     path = state_copy.transformation_history[path_start:]

#     # Update the operation features iteratively
#     for transformation, parameters in path:
#         if transformation == 'img2col':
#             # Ignore img2col because it's guaranteed to be already updated
#             continue
#         state_copy.operation_features = update_operation_features(state_copy, transformation, parameters)

#     return state_copy.operation_features


# def get_up_to_date_operation_features(state: OperationState) -> OperationFeatures:
#     """Get the up-to-date operation features.

#     Args:
#         state (OperationState): The current state.

#     Returns:
#         OperationFeatures: The up-to-date operation features.
#     """
#     if cfg.update_op_features:
#         # Features already updated
#         return state.operation_features

#     return update_operation_features_from_scratch(state)


# ================================================ Private functions ================================================


def __formula_str_to_list(formula: str):
    """
    Turns assignement formula to a list of (index, factor)
    Example:
        formula = "%x1 - %x2 + %x3 * 5 - %x5 * 3"
        return [('%x1', 1), ('%x2', -1), ('%x3', 5), ('%x5', -3)]

    Args:
        formula (str): the formula as a string input

    Returns:
        list: list of (index, factor) pairs
    """
    formula = formula + ' +'
    terms = formula.split(' ')

    running_factor = 1
    running_term = None

    save = []

    for term in terms:

        if term.startswith('%'):
            running_term = term
        elif term == '+':
            save.append((running_term, running_factor))
            running_factor = 1
        elif term == '-':
            save.append((running_term, running_factor))
            running_factor = -1
        elif term.isnumeric():
            running_factor *= int(term)

    if save[0][0] is None:
        save = save[1:]

    return save


def __remove_duplicate_args(args: list[str], shapes: list[str]):
    """Removes duplicate pairs from the list of paired arguments with shapes
    Args:
        args (list[str]): list of arguments
        shapes (list[str]): list of shapes

    Returns:
        list[str]: list of arguments without duplicates
        list[str]: list of shapes without duplicates
    """
    args_shapes = list(zip(args, shapes))
    seen = set()
    result = []
    for item in args_shapes:
        if item not in seen:
            seen.add(item)
            result.append(item)

    args = [x for (x, _) in result]
    shapes = [x for (_, x) in result]
    return args, shapes


def __function_wrapper(operation: str, maps: Optional[str] = None):
    """Wraps the operation line in a function in order to be able to lower into loops

    Args:
        operation (str): the operation line to be wrapped
        maps (Optional[str], optional): the affine maps. Defaults to None.

    Returns:
        str: the wrapped operation
    """
    ins_outs_pattern = r"(?:ins|outs)\s*\(([^())]+)\)"
    fields: list[str] = re.findall(ins_outs_pattern, operation)

    args: list[str] = []
    shapes: list[str] = []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    out_shape = shapes[-1]

    args, shapes = __remove_duplicate_args(args, shapes)

    args_str = ', '.join([f'{arg}: {shape}' for (arg, shape) in zip(args, shapes)])

    if maps is None:
        wrapped_operation = (
            f"func.func @func_call({args_str}) -> {out_shape} {{\n"
            f"  %ret = {operation}\n"
            f"  return %ret : {out_shape}\n"
            "}"
        )
    else:
        wrapped_operation = (
            f"{maps}\n"
            f"func.func @func_call({args_str}) -> {out_shape} {{\n"
            f"  %ret = {operation}\n"
            f"  return %ret : {out_shape}\n"
            "}"
        )

    return wrapped_operation


def __lower_linalg_to_loops(mlir_code: str, tmp_file_path: str):
    """
    Lower Linalg dialect code to Affine dialect

    Args:
        mlir_code (str): the MLIR code to be lowered to Affine dialect
        tmp_file_path (str): the temporary file to write the MLIR code to

    Returns:
        Optional[str]: the lowered code with affine dialect
    """
    # Write the MLIR code to a temporary file
    with open(tmp_file_path, "w") as file:
        file.write(mlir_code)

    # Lower the Linalg dialect code to Affine dialect
    out = os.popen(f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims --one-shot-bufferize=bufferize-function-boundaries --finalizing-bufferize --buffer-deallocation-pipeline --convert-linalg-to-affine-loops {tmp_file_path}").read()

    if out != '':
        return out
    else:
        return None


def __extract_bench_features_from_ast_result(bench_name: str, raw_ast_info: str, root_execution_time: int):
    """Extracts benchmark features from the code's AST result and execution time.

    Args:
        bench_name (str): the benchmark name
        raw_ast_info (str): the raw AST information
        root_execution_time (int): the root execution time
        execution_time (int): the execution time

    Returns:
        BenchmarkFeatures: extracted benchmark features
    """
    info, full_code = raw_ast_info.split("########################################")
    operations_lines, _ = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    ops_tags = []
    operations = {}
    for operation_block in operations_blocks:
        raw_operation, rest = operation_block.split("#START_VECTORIZABLE")
        operation_type = __get_operation_type(raw_operation)
        if operation_type is None:
            continue

        nested_loops = []
        op_count = {}
        load_data: list[list[str]] = []
        store_data: list[str] = []

        vectorizable_str, rest = rest.split("#START_NESTED_LOOPS")
        assert vectorizable_str.strip() in ["true", "false"], f"Vectorizable string is not valid: {vectorizable_str}"
        vectorizable = vectorizable_str.strip() == "true"

        nested_loops_str, rest = rest.split("#START_LOAD_DATA")
        loop_args = []
        for nested_loop_str in nested_loops_str.strip().split("\n"):
            if not nested_loop_str:
                continue
            arg, low, high, step, iter = nested_loop_str.strip().split(" ")
            nested_loops.append(NestedLoopFeatures(
                arg=f'%{arg}',
                lower_bound=int(low),
                upper_bound=int(high),
                step=int(step),
                iterator_type=iter
            ))
            loop_args.append(arg)

        loads_data_str, rest = rest.split("#START_STORE_DATA")
        for loop_arg in loop_args:
            loads_data_str = loads_data_str.replace(loop_arg, f'%{loop_arg}')
        for load_data_str in loads_data_str.strip().split("\n"):
            if not load_data_str:
                continue
            load_data.append(load_data_str.split(", "))

        store_data_str, rest = rest.split("#START_OP_COUNT")
        for loop_arg in loop_args:
            store_data_str = store_data_str.replace(loop_arg, f'%{loop_arg}')
        store_data_list = store_data_str.strip().split("\n")
        assert len(store_data_list) == 1, f"Store data list is not of length 1: {store_data_list}"
        store_data = store_data_list[0].split(", ")

        ops_count_str, rest = rest.split("#START_TAG")
        for op_count_str in ops_count_str.strip().split("\n"):
            op, count = op_count_str.strip().split(" ")
            op_count[op] = int(count)

        operation_tag = rest.strip().split("\n")[0]
        ops_tags.append(operation_tag)
        operations[operation_tag] = OperationFeatures(
            raw_operation=raw_operation,
            operation_type=operation_type,
            op_count=op_count,
            load_data=load_data,
            store_data=store_data,
            nested_loops=nested_loops,
            vectorizable=vectorizable
        )

    return BenchmarkFeatures(
        bench_name=bench_name,
        code=full_code,
        operation_tags=ops_tags,
        operations=operations,
        root_exec_time=root_execution_time,
    )


def __get_operation_type(raw_operation: str) -> Optional[OperationType]:
    """Get the operation type from the raw operation string.

    Args:
        raw_operation (str): The raw operation string.

    Returns:
        str: The operation type.
    """
    if 'linalg.matmul' in raw_operation:
        return 'matmul'
    elif 'linalg.conv' in raw_operation:
        return 'conv_2d'
    elif 'pooling' in raw_operation:
        return 'pooling'
    elif 'linalg.add' in raw_operation:
        return 'add'
    elif 'linalg.generic' in raw_operation:
        return 'generic'
    else:
        return None
