# pylint: skip-file

import os
import re
import numpy as np
from copy import copy
from dotenv import load_dotenv
load_dotenv()

from utils.consts import (
    MAX_NUM_STORES_LOADS,
    MAX_NUM_LOOPS,
    MAX_NUM_LOAD_STORE_DIM
)



def remove_duplicate_args(args, shapes):
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


def function_wrapper(operation, maps=None):
    """
    Wraps the operation line in a function in order to be able to lower into loops
    """
    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    args, shapes = [], []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args   += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    out_shape = shapes[-1]

    args, shapes = remove_duplicate_args(args, shapes)

    args_str = ', '.join([f'{arg}: {shape}' for (arg, shape) in zip(args, shapes)])

    if maps is None:
        wrapped_operation = \
        f"func.func @func_call({args_str}) -> {out_shape} {{\n" + \
        f"  %ret = {operation}\n" + \
        f"  return %ret : {out_shape}\n" + \
        "}"
    else:
        wrapped_operation = \
        f"{maps}\n" + \
        f"func.func @func_call({args_str}) -> {out_shape} {{\n" + \
        f"  %ret = {operation}\n" + \
        f"  return %ret : {out_shape}\n" + \
        "}"

    return wrapped_operation


def transform_wrapper(operation, maps=None):

    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    args, shapes = [], []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args   += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

    args, shapes = remove_duplicate_args(args, shapes)

    # print(args, shapes)

    #############################################################
    # consts:
    dims = []
    unique_dims = set()
    for shape in shapes:
        if shape.startswith("tensor"):
            arg_dims = list(map(int, re.findall(r'\d+', shape[7:-5])))
            dims.append( arg_dims )
            unique_dims = unique_dims.union(arg_dims)
        else: # shape == "f32"
            dims.append( -1 )
            unique_dims = unique_dims.union([-1])

    unique_dims = sorted(list(unique_dims))

    # print(unique_dims)

    consts_snippet = ""
    for dim in unique_dims:
        if dim != -1:
            consts_snippet += f"  %c{dim} = arith.constant {dim} : index\n"

    #############################################################
    # allocations:

    allocations_snippet = ""

    for arg, shape, arg_dims in zip(args, shapes, dims):
        # print(arg, shape, arg_dims)
        if shape.startswith("tensor"):
            n = shape.count("x")
            temp_shape = "tensor<" + "?x"*n + shape[-4:] # f32> or i64> ir i32>
            alloc_params = ", ".join([f"%c{dim}" for dim in arg_dims])
            allocations_snippet += f"  {arg}_temp = bufferization.alloc_tensor({alloc_params}) : {temp_shape}\n"
            allocations_snippet += f"  {arg} = tensor.cast {arg}_temp : {temp_shape} to {shape}\n"
        else:
            # print(arg, shape, arg_dims)
            allocations_snippet += f"  {arg} = arith.constant 1.00000e+00 : f32\n"

    # print(allocations_snippet)

    #############################################################
    # function call:

    function_call_snippet = f"  %ret_arg = func.call @func_call({', '.join(args)}) : ({', '.join(shapes)}) -> ({shapes[-1]})"

    #############################################################
    # All code:

    code = ""
    if maps is not None:
        code += f"{maps}\n"
    code += 'module attributes {torch.debug_module_name = "Net"} {\n'
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printFlops(f64)\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += "\n"
    code += "\n"
    code +=f"func.func @matmul() -> {shapes[-1]}{{\n"
    code += "\n"
    code += "%val = arith.constant 2.00000e+00 : f32\n"
    code += "%zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"

    # code +=f"%out = bufferization.alloc_tensor() : tensor<{N}x{K}xf32>\n"
    # code +=f"%A = linalg.fill ins(%val : f32) outs(%out : tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>\n"
    for arg, shape, arg_dims in zip(args, shapes, dims):
        if shape != 'f32':
            tmp_arg = f'%tmp_{arg[1:]}'
            code +=f"{tmp_arg} = bufferization.alloc_tensor() : {shape}\n"
            code +=f"{arg} = linalg.fill ins(%val : f32) outs({tmp_arg} : {shape}) -> {shape}\n"
        else:
            code +=f"{arg} = arith.constant 2.00000e+00 : f32\n"

    code += "\n"
    code += "%t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "\n"

    # code +=f"%D = linalg.matmul ins(%A, %B: tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%C: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>\n"
    code += f"%return_arg = {operation}"

    code += "\n"
    code += "%t = func.call @nanoTime() : () -> (i64)\n"
    code += "%delta = arith.subi %t, %t0 : i64\n"
    code += "%fp = arith.uitofp %delta : i64 to f64\n"
    code += "// func.call @printFlops(%fp) : (f64) -> ()\n"
    code += "func.call @printI64(%delta) : (i64) -> ()\n"
    code += "func.call @printNewline() : () -> ()\n"
    code += "\n"
    code +=f"return %return_arg : {shapes[-1]} \n"
    code += "}\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code +=f"    %outputmain = func.call @matmul() : () -> {shapes[-1]}\n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"
    code += "}\n"

    return code


def lower_linalg_to_loops(mlir_code, tmp_file):
    """
    Lower Linalg dialect code to Affine dialect
    """
    # command = f"echo '{mlir_code}' | {os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims --linalg-bufferize --convert-linalg-to-affine-loops /dev/stdin"
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"
    with open(tmp_file, "w") as file:
        file.write(mlir_code)

#     one-shot-bufferize{bufferize-function-boundaries}
# func.func(
#     finalizing-bufferize
# )
# buffer-deallocation-pipeline

    out = os.popen(f"{os.getenv('LLVM_BUILD_PATH')}/bin/mlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims --one-shot-bufferize=bufferize-function-boundaries --finalizing-bufferize --buffer-deallocation-pipeline --convert-linalg-to-affine-loops {tmp_file}").read()

    if out != '':
        return out
    else:
        return None


def get_nested_loops_data(loops):

    lines = loops.split('\n') if loops else []

    loops_detailed = {}
    loops_detailed["nested_loops"] = []
    loops_detailed["op_count"] = {'+':0, '-':0, '*':0, '/':0, 'exp':0,}
    loops_detailed["load_data"] = []
    loops_detailed["store_data"] = []

    maps = {}
    args_of_loops = []
    args_of_map = {}

    for line in lines:

        if "affine_map" in line:
            map_name, map_function = line.strip().split(' = ')
            map_function = map_function.split(' -> ')[1][1:-2]
            maps[map_name] = map_function

        elif "affine.apply" in line:
            new_op, _, _, *map_name__args = line.strip().split(' ')
            map_name__args = ' '.join(map_name__args)
            s = map_name__args.index('(')
            map_name, args = map_name__args[:s], map_name__args[s+1:-1].split(', ')
            mapping_string = copy(maps[map_name])
            for i in range(len(args)):
                mapping_string = mapping_string.replace(f'd{i}', args[i])
            # print(new_op, map_name, args, maps[map_name], mapping_string)
            args_of_map[new_op] = mapping_string

        elif "affine.for" in line:
            _, arg, _, lower, _, upper, _ = line.strip().split(' ')
            # print(arg, lower, upper)
            loops_detailed["nested_loops"].append((arg, int(lower), int(upper), 1))
            args_of_loops.append(arg)

        elif "affine.load" in line:
            # print(line.strip().split(' ')[:-2])
            new_op, _, _, *alloc = line.strip().split(' ')[:-2]
            alloc = ' '.join(alloc)
            args = alloc.split('[')[1][:-1].split(', ')

            for i in range(len(args)):
                if args[i] in args_of_map:
                    args[i] = args_of_map[args[i]]

            loops_detailed["load_data"].append(args)

        elif "arith.addf" in line:loops_detailed["op_count"]['+'] += 1
        elif "arith.mulf" in line:loops_detailed["op_count"]['*'] += 1
        elif "arith.subf" in line:loops_detailed["op_count"]['-'] += 1
        elif "arith.divf" in line:loops_detailed["op_count"]['/'] += 1
        elif "math.exp" in line:loops_detailed["op_count"]['exp'] += 1

    return loops_detailed


def formula_str_to_list(formula):
    """
    Turns assignement formula to a list of (index, factor)
    Example:
        formula = "%x1 - %x2 + %x3 * 5 - %x5 * 3"
        return [('%x1', 1), ('%x2', -1), ('%x3', 5), ('%x5', -3)]
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

    if save[0][0] == None:
        save = save[1:]

    return save


def build_nested_loops_feature_vector(loops_data):

    indices = [arg for (arg, lower_bound, upper_bound, step, iter) in loops_data['nested_loops']]
    indices_dim = {arg: i for (i, arg) in enumerate(indices)}

    # Nested loop features: (upper/lower bounds, step)
    nested_loops = np.zeros((MAX_NUM_LOOPS,))
    for i, (arg, lower_bound, upper_bound, step, iter) in enumerate(loops_data['nested_loops']):
        if i == MAX_NUM_LOOPS:
            break
        nested_loops[i] = upper_bound

    # load access matrices:

    load_data = loops_data["load_data"]

    load_access_matrices = np.zeros((MAX_NUM_STORES_LOADS, MAX_NUM_LOAD_STORE_DIM, MAX_NUM_LOOPS), dtype=np.int16)

    for load_i, load in enumerate(load_data):
        dimensions_terms = [formula_str_to_list(term) for term in load]
        for m, dimension_term in enumerate(dimensions_terms):
            for index, factor in dimension_term:
                if index in indices_dim:
                    n = indices_dim[index]
                    load_access_matrices[load_i, m, n] = factor

    # load access matrices:
    store_data = loops_data["store_data"]

    store_access_matrices = np.zeros((MAX_NUM_LOAD_STORE_DIM, MAX_NUM_LOOPS), dtype=np.int16)

    dimensions_terms = [formula_str_to_list(term) for term in store_data]
    for m, dimension_term in enumerate(dimensions_terms):
        for index, factor in dimension_term:
            n = indices_dim[index]
            store_access_matrices[m, n] = factor


    # Operations count:
    operations_count =  np.array(list(loops_data["op_count"].values()))

    # Feature vector:
    nested_loops = nested_loops.reshape(-1)
    load_access_matrices = load_access_matrices.reshape(-1)
    store_access_matrices = store_access_matrices.reshape(-1)

    # print('   ', nested_loops.shape, load_access_matrices.shape, store_access_matrices.shape, operations_count.shape)
    feature_vector = np.concatenate((nested_loops, load_access_matrices, store_access_matrices, operations_count))

    return feature_vector


class AutoScheduleOperation:
    def __init__(self, operation):
        self.operation = operation
        self.wrapped_operation = function_wrapper(operation)
        self.lowered_operation = lower_linalg_to_loops(self.wrapped_operation)
        self.loops_data = get_nested_loops_data(self.lowered_operation)
        self.transform_wrapped_operation = transform_wrapper(operation, self.wrapped_operation)
        self.transformed_code = operation

    def __str__(self):
        return self.operation

    def get_feature_vector(self):
        return build_nested_loops_feature_vector(self.loops_data)
