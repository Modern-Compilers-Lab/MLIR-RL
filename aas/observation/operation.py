from dataclasses import dataclass
from typing import Literal, Optional
import os
import subprocess
import re
from copy import copy


# ================================================ Oberservable Features ================================================


@dataclass
class NestedLoopFeatures:
    """Dataclass to store the nested loops features data."""
    arg: str
    """The argument representing the loop iterator."""
    lower_bound: int
    """The lower bound of the loop."""
    upper_bound: int
    """The upper bound of the loop."""
    step: int
    """The loop step."""
    iterator_type: Literal["parallel", "reduction"]
    """The type of the loop iterator."""

    def __repr__(self):
        return f"NestedLoopFeatures(arg={self.arg}, lower_bound={self.lower_bound}, upper_bound={self.upper_bound}, step={self.step}, iterator_type={self.iterator_type})"


@dataclass
class OperationFeatures:
    """Dataclass to store the operation features data."""
    operation_type: Literal['matmul', 'conv_2d', 'conv_2d+img2col', 'pooling', 'add', 'generic', 'unknown']
    """The type of the operation (matmul, conv_2d, pooling, add, generic, unknown)."""
    op_count: dict[str, int]
    """Number of arithmetic operations in the operation."""
    op_iter_space_size: int
    """The iteration space size of the operation."""
    nested_loops: list[NestedLoopFeatures]
    """List of nested loops where each loop is represented by the NestedLoopFeatures dataclass."""
    load_data: list[list[str]]
    """List of load accesses where each load is represented by the list of access arguments."""
    store_data: list[list[str]]
    """List of store accesses where each store is represented by the list of access arguments."""
    vectorizable: bool = True
    """Whether the operation is vectorizable or not. Defaults to True."""

    def __repr__(self):
        return f"OperationFeatures(operation_type={self.operation_type}, op_count={self.op_count}, op_iter_space_size={self.op_iter_space_size}, nested_loops={self.nested_loops}, load_data={self.load_data}, store_data={self.store_data})"


# ================================================ Public functions ================================================


def extract_op_features_from_code(code: str, operation_tag: str):
    """Extracts operation features from the code and operation tag.

    Args:
        code (str): the code to extract features from
        operation_tag (str): the operation tag

    Returns:
        OperationFeatures: extracted operation features
    """
    result = subprocess.run(
        f'{os.getenv("AST_DUMPER_BIN_PATH")} -',
        shell=True,
        input=code.encode('utf-8'),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    raw_ast_info = result.stdout.decode('utf-8')

    return __extract_op_features_from_ast_result(raw_ast_info, operation_tag)


def formula_str_to_list(formula: str):
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


def extract_op_features_from_affine_code(raw_operation: str, tmp_file_path: str):
    """Get operation features from the raw operation.

    Args:
        raw_operation (str): the raw operation
        tmp_file_path (str): the temporary file path to write the operation to

    Returns:
        OperationFeatures: operation features contained in the raw operation
    """
    # Get operation type
    if 'linalg.matmul' in raw_operation:
        operation_type = 'matmul'
    elif 'linalg.conv' in raw_operation:
        operation_type = 'conv_2d'
    elif 'pooling' in raw_operation:
        operation_type = 'pooling'
    elif 'linalg.add' in raw_operation:
        operation_type = 'add'
    elif 'linalg.generic' in raw_operation:
        operation_type = 'generic'
    else:
        operation_type = 'unknown'

    # Get code as affine loops
    wrapped_operation = __function_wrapper(raw_operation)
    loops = __lower_linalg_to_loops(wrapped_operation, tmp_file_path)
    lines = loops.split('\n') if loops else []

    # Build op features
    nested_loops = []
    op_iter_space_size = 1
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
            # print(new_op, map_name, args, maps[map_name], mapping_string)
            args_of_map[new_op] = mapping_string

        elif "affine.for" in line:
            _, arg, _, lower, _, upper, _ = line.strip().split(' ')
            # print(arg, lower, upper)
            # TODO: handle iterator types better
            lb = int(lower)
            ub = int(upper)
            nested_loops.append(
                NestedLoopFeatures(
                    arg=arg,
                    lower_bound=lb,
                    upper_bound=ub,
                    step=1,
                    iterator_type='parallel'
                )
            )
            args_of_loops.append(arg)
            op_iter_space_size *= ub - lb

        elif "affine.load" in line:
            # print(line.strip().split(' ')[:-2])
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
        operation_type=operation_type,
        op_iter_space_size=op_iter_space_size,
        op_count=op_count,
        nested_loops=nested_loops,
        load_data=load_data,
        store_data=store_data
    )


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


def get_operation_type(raw_operation: str) -> Optional[str]:
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


# ================================================ Private functions ================================================

def __extract_op_features_from_ast_result(raw_ast_info: str, operation_tag: str):
    """Extract operation features from the raw ast info.

    Args:
        raw_ast_info (str): the raw ast info
        operation_tag (str): the operation tag

    Returns:
        OperationFeatures: extracted operation features
    """
    info, _ = raw_ast_info.split("########################################")

    operations_lines, _ = info.split('#BEGIN_GRAPH')

    operations_blocks = operations_lines.split('#START_OPERATION')
    operations_blocks = [block.strip() for block in operations_blocks if block]

    operation_block_id = -1
    for i, operation_block_str in enumerate(operations_blocks):
        _, rest = operation_block_str.split("#START_TAG")
        block_operation_tag = rest.strip().split("\n")[0]
        if block_operation_tag == operation_tag:
            operation_block_id = i
            break
    if operation_block_id == -1:
        # raise Exception(f"Operation tag '{operation_tag}' not found in the code.")
        return None

    operation_block = operations_blocks[operation_block_id]

    raw_operation, rest = operation_block.split("#START_VECTORIZABLE")
    operation_type = get_operation_type(raw_operation)

    nested_loops = []
    op_count = {}
    load_data: list[list[str]] = []
    store_data: list[str] = []

    vectorizable_str, rest = rest.split("#START_NESTED_LOOPS")
    assert vectorizable_str.strip() in ["true", "false"], f"Vectorizable string is not valid: {vectorizable_str}"
    vectorizable = vectorizable_str.strip() == "true"

    nested_loops_str, rest = rest.split("#START_LOAD_DATA")
    op_iter_space_size = 1
    for nested_loop_str in nested_loops_str.strip().split("\n"):
        if not nested_loop_str:
            continue
        arg, low, high, step, iter = nested_loop_str.strip().split(" ")
        nested_loop = NestedLoopFeatures(
            arg=f'%{arg}',
            lower_bound=int(low),
            upper_bound=int(high),
            step=int(step),
            iterator_type=iter
        )
        op_iter_space_size *= nested_loop.upper_bound - nested_loop.lower_bound
        nested_loops.append(nested_loop)

    loads_data_str, rest = rest.split("#START_STORE_DATA")
    loads_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', loads_data_str)
    for load_data_str in loads_data_str.strip().split("\n"):
        if not load_data_str:
            continue
        load_data.append(load_data_str.split(", "))

    store_data_str, rest = rest.split("#START_OP_COUNT")
    store_data_str = re.sub(r'd\d+', lambda m: f'%{m.group()}', store_data_str)
    store_data_list = store_data_str.strip().split("\n")
    assert len(store_data_list) == 1, f"Store data list is not of length 1: {store_data_list}"
    store_data = store_data_list[0].split(", ")

    ops_count_str, rest = rest.split("#START_TAG")
    for op_count_str in ops_count_str.strip().split("\n"):
        op, count = op_count_str.strip().split(" ")
        op_count[op] = int(count)

    operation_tag = rest.strip().split("\n")[0]
    return OperationFeatures(
        operation_type=operation_type,
        op_count=op_count,
        op_iter_space_size=op_iter_space_size,
        nested_loops=nested_loops,
        load_data=load_data,
        store_data=store_data,
        vectorizable=vectorizable
    )


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
