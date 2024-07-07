# pylint: skip-file

from utils.observation_utils import AutoScheduleOperation
from random import randint, choice, shuffle
from tqdm import tqdm
import json

from numpy import median

from utils.observation_utils import *
from utils.transform_utils import *

sys.path.append('./llvm-project/build/tools/mlir/python_packages/mlir_core')
from mlir.ir import Context, Module
from mlir import ir


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


def transform_wrapper(operation):

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
    code += "\n"

    return code




def get_linalg_operations(mlir_code):
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
        # print(mlir_linalg)
        root = Module.parse(mlir_code, managed_ctx)

        main_blocks = [op for op in root.body.operations if "func.func" in str(op)]
        assert len(main_blocks) == 1, 'Mlir code containing multiple functions'
        
        main_operations = main_blocks[0].body.blocks[0].operations
        
        linalg_operations = [op for op in main_operations if "linalg." in str(op)]
        linalg_operations = [op for op in linalg_operations if not "linalg.fill ins(%cst : f32)" in str(op)]
        linalg_operations = [' '.join(str(op).split(' ')[2:]) if str(op).startswith("%") else str(op) for op in linalg_operations ]
        
        
        return linalg_operations


with open('generated_data/vision_linalg_operations.json', 'r') as file:
    linalg_operations = json.load(file)['0']

exec_time = None
data = []
# with open('generated_data/conv_2d_vision_operations.json', 'r') as file:
#     data = json.load(file)

print('len(data):', len(data))
# data = []

tqdm_bar = tqdm(enumerate(linalg_operations), total=len(linalg_operations))
for i, mlir_linalg in tqdm_bar:

    tqdm_bar.set_description(f'len(data) = {len(data)}')
    
    mlir_linalg = mlir_linalg.replace('arith.minimumf', 'arith.minf')
    mlir_linalg = mlir_linalg.replace('arith.maximumf', 'arith.maxf')
    
    if "%11 = arith.subi %6, %10 : index" in mlir_linalg:
        # print(mlir_linalg)
        continue
    
    
    
    try:
        linalg_operations = get_linalg_operations(mlir_linalg)
    except:
        continue
    
    for operation in linalg_operations:
        
        try:
        
            if operation in data:
                continue
            
            if not 'conv_2d' in operation:
                continue
            
            
            
            
            operation = operation.replace('666', str(choice([128, 256, 512])))
            # operation = operation.replace('666', '512')
            
            
            
            
            wrapped_operation = function_wrapper(operation)  
            loops = lower_linalg_to_loops(wrapped_operation)
            loops_data = get_nested_loops_data(loops)
            
            if loops_data == None:
                continue
            
            feature_vector = build_nested_loops_feature_vector(loops_data)
            transform_wrapped_operation = transform_wrapper(operation)
            
            
            exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 60)
            if exec_time is None:
                continue
            
            
            data.append([
                operation,
                {
                    'operation':operation,
                    'wrapped_operation':wrapped_operation,
                    'lowered_operation':loops,
                    'transform_wrapped_operation':transform_wrapped_operation,
                    'loops_data':loops_data,
                    'execution_time':exec_time
                }
            ])

        except:
            pass

        with open('generated_data/conv_2d_all_vision_operations.json', 'w') as file:
            json.dump(data, file)
            
        
        
    # if len(data) > 50:
    #     break

with open('generated_data/conv_2d_all_vision_operations.json', 'w') as file:
        json.dump(data, file)