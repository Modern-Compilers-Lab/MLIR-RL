# pylint: skip-file

import os
import re
import subprocess
import sys
import numpy as np

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

def conv_2d_fhwc_to_hwcf(operation):
    """
    transform `linalg.conv_2d_nhwc_fhwc` to `linalg.conv_2d_nhwc_hwcf` 
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

    input_shape, filter_shape, out_shape = [shape[7:-5].split('x') for shape in shapes]
    filter_shape = filter_shape[1:] + filter_shape[:1]
    shapes = [input_shape, filter_shape, out_shape]
    shapes = ["tensor<" + 'x'.join(shape) + "xf32>" for shape in shapes]



    # print("input_shape:\t", "tensor<" + 'x'.join(input_shape) + "xf32>")
    # print("filter_shape:\t", "tensor<" + 'x'.join(filter_shape) + "xf32>")
    # print("out_shape:\t", "tensor<" + 'x'.join(out_shape) + "xf32>")

    transposed_op = "linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(" + ', '.join(args[:2]) + " : " + ', '.join(shapes[:2]) + f") outs({args[-1]} : {shapes[-1]}) -> {shapes[-1]}"

    return transposed_op

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


def transform_wrapper(operation, wrapped_operation=None):

    if wrapped_operation == None:
        wrapped_operation = function_wrapper(operation)

    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    args, shapes = [], []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args   += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

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

    consts_snippet = ""
    for dim in unique_dims:
        if dim != -1:
            consts_snippet += f"  %c{dim} = arith.constant {dim} : index\n"

    #############################################################
    # allocations:

    allocations_snippet = ""

    for arg, shape, arg_dims in zip(args, shapes, dims):
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
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += "\n\n"
    code += wrapped_operation
    code += "\n\n"
    code += "func.func @main() -> i64 {\n"
    code +=    consts_snippet
    code +=    allocations_snippet
    code += "  %t0 = func.call @nanoTime() : () -> (i64)\n"
    code +=    function_call_snippet
    code += "  %t1 = func.call @nanoTime() : () -> (i64)\n"
    code += "  %delta = arith.subi %t1, %t0 : i64\n"
    code += "  return %delta : i64\n"
    code += "}"

    return code


def transform_wrapper_2(operation, wrapped_operation=None):

    if wrapped_operation == None:
        wrapped_operation = function_wrapper(operation)

    ins_outs_pattern = "(?:ins|outs)\s*\(([^())]+)\)"
    fields = re.findall(ins_outs_pattern, operation)

    args, shapes = [], []
    for field in fields:
        args_field, shapes_field = field.split(':')
        args   += args_field.split(',')
        shapes += shapes_field.split(',')

    args = [arg.strip() for arg in args]
    shapes = [shape.strip() for shape in shapes]

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

    consts_snippet = ""
    for dim in unique_dims:
        if dim != -1:
            consts_snippet += f"  %c{dim} = arith.constant {dim} : index\n"

    #############################################################
    # allocations:

    allocations_snippet = ""

    for arg, shape, arg_dims in zip(args, shapes, dims):
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
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += "\n\n"
    code += wrapped_operation
    code += "\n\n"
    code += "func.func @main() {\n"
    code +=    consts_snippet
    code +=    allocations_snippet
    code += "  %t0 = func.call @nanoTime() : () -> (i64)\n"
    code +=    function_call_snippet
    code += "  %t1 = func.call @nanoTime() : () -> (i64)\n"
    code += "  %delta = arith.subi %t1, %t0 : i64\n"
    code += "  func.call @printI64(%delta) : (i64) -> ()\n"
    code += "  return\n"
    code += "}"

    return code


def lower_linalg_to_loops(mlir_code):
    """
    Lower Linalg dialect code to Affine dialect
    """
    # command = f'echo "{mlir_code}" | /scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims --linalg-bufferize --convert-linalg-to-affine-loops /dev/stdin'
    # result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"
    with open(tmp_file, "w") as file:
        file.write(mlir_code)
    
    out = os.popen(f"""/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt --linalg-fuse-elementwise-ops --linalg-fold-unit-extent-dims --linalg-bufferize --convert-linalg-to-affine-loops {tmp_file}""").read()

    if out != '':
        return out
    else:
        return None


def get_nested_loops_data(mlir_code):
    """
    Get high level features from the nested loops in the mlir code
    Features:
        Loop Nest features:
            Lower bound, Higher bound, Step
        Assignment features:
            Load Access matrix
            Store Access matrix
            Operations count
    """
    with Context() as managed_ctx:

        managed_ctx.allow_unregistered_dialects = True
        
        root = Module.parse(mlir_code)
        main = root.body.operations[0].body.blocks[0]

        loops = []
        for i, op in enumerate(main.operations):
            # print(op, type(op), isinstance(op, ir.OpView))
            if 'affine.for ' in str(op):
                loops.append(op)
        
        if len(loops) == 0:
            print('No loops found')
            return None

        for loop in loops:
            loops_detailed = {}
            loops_detailed["nested_loops"] = []
            loops_detailed["body"] = []
            loops_detailed["op_count"] = {'+':0, '-':0, '*':0, '/':0, 'exp':0,}
            loops_detailed["load_data"] = []
            loops_detailed["store_data"] = []

            op = loop
            while True:
                
                lower_bound = int(str(op.attributes['lower_bound'])[18:-2])
                upper_bound = int(str(op.attributes['upper_bound'])[18:-2])
                step = int(str(op.attributes['step']).split(' ')[0])
                arg = str(op).split(' ')[1]

                loops_detailed["nested_loops"].append((arg, lower_bound, upper_bound, step))

                # if op.regions[0].blocks[0].operations[0].name == "affine.for":
                if "affine.for " in str(op.regions[0].blocks[0].operations[0]):
                    op = op.regions[0].blocks[0].operations[0]
                    
                else:
                    body = op.regions[0].blocks[0].operations
                    for op_ in body:
                        loops_detailed["body"].append(op_)
                    break

            
            map_vars = {} # Result variables of affine.apply
            
            # Parse the body of the nested loops:
            for op in loops_detailed['body']:
                # if op.name == 'affine.apply':
                if 'affine.apply' in str(op):
                    result_var = op.result.get_name()

                    # get the simplified formula
                    operands = [operand.get_name() for operand in op.operands]
                    affinemap = str(op.attributes[0].attr)[11:-1]
                    affinemap_left, affinemap_right = affinemap.split(' -> ')
                    affinemap_left, affinemap_right = affinemap_left[1:-1], affinemap_right[1:-1]
                    tmp_operands = affinemap_left.split(', ')

                    for true_operand, tmp_opernad in zip(operands, tmp_operands):
                        affinemap_right = affinemap_right.replace(tmp_opernad, true_operand)

                    str_formula = affinemap_right
                    map_vars[result_var] = str_formula

                # elif op.name == "affine.load":
                elif 'affine.load' in str(op):
                    result_var = op.result.get_name()
                    all_operands = [operand.get_name() for operand in op.operands]
                    load_index_vars = all_operands[1:]

                    # Replace map_var with their simplified formula
                    for i, var in enumerate(load_index_vars):
                        if var in map_vars:
                            load_index_vars[i] = map_vars[var]

                    loops_detailed["load_data"].append(load_index_vars)

                # elif op.name == 'affine.store':
                elif 'affine.store' in str(op):
                    all_operands = [operand.get_name() for operand in op.operands]
                    store_index_vars = all_operands[2:]
                    
                    loops_detailed["store_data"] = store_index_vars

                # elif op.name == "arith.addf":loops_detailed["op_count"]['+'] += 1
                # elif op.name == "arith.mulf":loops_detailed["op_count"]['*'] += 1
                # elif op.name == "arith.subf":loops_detailed["op_count"]['-'] += 1
                # elif op.name == "arith.divf":loops_detailed["op_count"]['/'] += 1
                # elif op.name == "math.exp":loops_detailed["op_count"]['exp'] += 1
                elif "arith.addf" in str(op):loops_detailed["op_count"]['+'] += 1
                elif "arith.mulf" in str(op):loops_detailed["op_count"]['*'] += 1
                elif "arith.subf" in str(op):loops_detailed["op_count"]['-'] += 1
                elif "arith.divf" in str(op):loops_detailed["op_count"]['/'] += 1
                elif "math.exp" in str(op):loops_detailed["op_count"]['exp'] += 1

    del loops_detailed['body']
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
    
    indices = [arg for (arg, lower_bound, upper_bound, step) in loops_data['nested_loops']]
    indices_dim = {arg:i for (i, arg) in enumerate(indices)}

    # Nested loop features: (upper/lower bounds, step)
    nested_loops = np.zeros((MAX_NUM_LOOPS))
    for i, (arg, lower_bound, upper_bound, step) in enumerate(loops_data['nested_loops']):
        if i == MAX_NUM_LOOPS:break
        nested_loops[i] = upper_bound

    # load access matrices:
    
    load_data = loops_data["load_data"]

    load_access_matrices = np.zeros((MAX_NUM_STORES_LOADS, MAX_NUM_LOAD_STORE_DIM, MAX_NUM_LOOPS), dtype=np.int16)

    for load_i, load in enumerate(load_data):
        dimensions_terms = [formula_str_to_list(term) for term in load]
        for m, dimension_term in enumerate(dimensions_terms):
            for index, factor in dimension_term:
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


def build_matmul_transform_code(loops_data):
    
    N, M, K = [ub for (arg, lb, ub, s) in loops_data["nested_loops"]]
    
    code = ""
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }\n"
    code += "func.func private @printFlops(f64)\n"
    code += "func.func private @printI64(i64)\n"
    code += "func.func private @printNewline()\n"
    code += "func.func private @printMemrefF32(tensor<*xf32>)\n"
    code += "\n"
    code += "\n"
    code +=f"func.func @matmul() -> tensor<{N}x{M}xf32>{{\n"
    code += "\n"
    code += "%val = arith.constant 2.00000e+00 : f32\n"
    code += "%zero = arith.constant 0.00000e+00 : f32\n"
    code += "\n"
    code +=f"%out = bufferization.alloc_tensor() : tensor<{N}x{K}xf32>\n"
    code +=f"%A = linalg.fill ins(%val : f32) outs(%out : tensor<{N}x{K}xf32>) -> tensor<{N}x{K}xf32>\n"
    code +=f"%out1 = bufferization.alloc_tensor() : tensor<{K}x{M}xf32>\n"
    code +=f"%B = linalg.fill ins(%val : f32) outs(%out1 : tensor<{K}x{M}xf32>) -> tensor<{K}x{M}xf32>\n"
    code +=f"%out2 = bufferization.alloc_tensor() : tensor<{N}x{M}xf32>\n"
    code +=f"%C = linalg.fill ins(%zero : f32) outs(%out2 : tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>\n"
    code += "\n"
    code += "%t0 = func.call @nanoTime() : () -> (i64)\n"
    code += "\n"
    code +=f"%D = linalg.matmul ins(%A, %B: tensor<{N}x{K}xf32>, tensor<{K}x{M}xf32>) outs(%C: tensor<{N}x{M}xf32>) -> tensor<{N}x{M}xf32>\n"
    code += "\n"
    code += "%t = func.call @nanoTime() : () -> (i64)\n"
    code += "%delta = arith.subi %t, %t0 : i64\n"
    code += "%fp = arith.uitofp %delta : i64 to f64\n"
    code += "// func.call @printFlops(%fp) : (f64) -> ()\n"
    code += "func.call @printI64(%delta) : (i64) -> ()\n"
    code += "func.call @printNewline() : () -> ()\n"
    code += "\n"
    code +=f"return %D : tensor<{N}x{M}xf32> \n"
    code += "}\n"
    code += "\n"
    code += "func.func @main(){\n"
    code += "    %c1 = arith.constant 1: index\n"
    code += "    %c0 = arith.constant 0 : index\n"
    code += "    %n = arith.constant 2: index\n"
    code += "    scf.for %i = %c0 to %n step %c1 {\n"
    code +=f"    %outputmain = func.call @matmul() : () -> tensor<{N}x{M}xf32>\n"
    code += "    }\n"
    code += "    return\n"
    code += "}\n"
    code += "\n"
    
    return code


def build_conv2d_transform_code(loops_data):
    
    print(loops_data["nested_loops"])
    
    code = ""
    code += "func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }"
    code += "func.func private @printI64(i64)"
    code += "func.func private @printNewline()"
    code += ""
    code += ""
    code += "func.func @matmul() -> tensor<32x112x112x64xf32>{"
    code += ""
    code += "  %val = arith.constant 2.00000e+00 : f32"
    code += "  %zero = arith.constant 0.00000e+00 : f32"
    code += ""
    code += "  %out = bufferization.alloc_tensor() : tensor<32x230x230x3xf32>"
    code += "  %input = linalg.fill ins(%val : f32) outs(%out : tensor<32x230x230x3xf32>) -> tensor<32x230x230x3xf32>"
    code += "  %out1 = bufferization.alloc_tensor() : tensor<7x7x3x64xf32>"
    code += "  %filter = linalg.fill ins(%val : f32) outs(%out1 : tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>"
    code += "  %out2 = bufferization.alloc_tensor() : tensor<32x112x112x64xf32>"
    code += "  %init = linalg.fill ins(%zero : f32) outs(%out2 : tensor<32x112x112x64xf32>) -> tensor<32x112x112x64xf32>"
    code += ""
    code += "  %t0 = func.call @nanoTime() : () -> (i64)"
    code += ""
    code += "  %D = linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins (%input, %filter: tensor<32x230x230x3xf32>, tensor<7x7x3x64xf32>) outs (%init: tensor<32x112x112x64xf32>) -> tensor<32x112x112x64xf32>"
    code += "  "
    code += "  %t = func.call @nanoTime() : () -> (i64)"
    code += "  %delta = arith.subi %t, %t0 : i64"
    code += "  func.call @printI64(%delta) : (i64) -> ()"
    code += "  func.call @printNewline() : () -> ()"
    code += "  "
    code += "  return %D : tensor<32x112x112x64xf32> "
    code += "}"
    code += ""
    code += "func.func @main(){"
    code += "    %c1 = arith.constant 1: index"
    code += "    %c0 = arith.constant 0 : index"
    code += "    %n = arith.constant 2: index"
    code += "    scf.for %i = %c0 to %n step %c1 {"
    code += "      %outputmain = func.call @matmul() : () -> tensor<32x112x112x64xf32>"
    code += "    }"
    code += "    return"
    code += "}"



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


if __name__ == "__main__":
    operation = "linalg.matmul ins(%arg0, %arg1: tensor<1200x1500xf32>, tensor<1500x1000xf32>) outs(%arg2: tensor<1200x1000xf32>) -> tensor<1200x1000xf32>"
    print(operation)
    print('-'*50)

    wrapped_operation = function_wrapper(operation)
    print(wrapped_operation)
    print('-'*50)

    loops = lower_linalg_to_loops(wrapped_operation)
    # print(loops)
    # print('-'*50)

    loops_data = get_nested_loops_data(loops)
    print(loops_data)
    print('-'*50)

    feature_vector = build_nested_loops_feature_vector(loops_data)
    # print(feature_vector)
    
    print(build_matmul_transform_code(loops_data))

