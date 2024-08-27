from utils.observation_utils import function_wrapper, lower_linalg_to_loops
from utils.transform_utils import evaluate_code_with_timeout
import json, re
from tqdm import tqdm
from copy import copy
import numpy as np

def get_nested_loops_data(loops):
    
    lines = loops.split('\n')

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




# with open('collected_operations.json', 'r') as file:
#     operations = json.load(file)    
# matmuls = operations['matmul']


BS = 256

matmuls = [
    [(BS, 2048), (2048, 1000)],
    [(BS, 1280), (1280, 1000)],
    [(BS, 1536), (1536, 1000)],
    [(BS, 1408), (1408, 1000)],
    [(BS, 768), (768, 768)],
    [(BS, 1024), (1024, 1024)],
    [(BS, 768), (768, 3072)],
    [(BS, 256), (256, 128)],
    [(BS, 4096), (4096, 1024)],
    [(BS, 1536), (1536, 4096)],
    [(BS, 768), (768, 2)],
    [(BS, 2048), (2048, 2048)],
    [(BS, 128), (128, 256)],
    [(BS, 256), (256, 512)],
    [(BS, 512), (512, 1024)],
]

conv_2ds = [
    [(BS, 14, 14, 256), (3, 3, 256, 256), 1],
    [(BS, 14, 14, 256), (1, 1, 256, 1024), 1],
    [(BS, 28, 28, 128), (3, 3, 128, 128), 1],
    [(BS, 28, 28, 128), (1, 1, 128, 512), 1],
    [(BS, 28, 28, 512), (1, 1, 512, 128), 1],
    [(BS, 14, 14, 256), (1, 1, 256, 1024), 1],
    [(BS, 14, 14, 128), (3, 3, 128, 32), 1],
    [(BS, 7, 7, 128), (3, 3, 128, 32), 1],
    [(BS, 16, 16, 256), (3, 3, 256, 256), 1],
    [(BS, 14, 14, 576), (1, 1, 576, 576), 1],
    [(BS, 14, 14, 896), (3, 3, 896, 896), 1],
    [(BS, 14, 14, 384), (7, 7, 384, 384), 1],
    [(BS, 28, 28, 128), (3, 3, 128, 32), 1],
    [(BS, 14, 14, 336), (1, 1, 336, 336), 1],
    [(BS, 56, 56, 64), (3, 3, 64, 64), 1],
    [(BS, 28, 28, 448), (1, 1, 448, 448), 1],
    [(BS, 56, 56, 64), (1, 1, 64, 256), 1],
    [(BS, 128, 128, 16), (7, 7, 16, 8), 2],
    [(BS, 64, 64, 64), (3, 3, 64, 16), 1],
    [(BS, 32, 32, 32), (7, 7, 32, 256), 2],
    [(BS, 230, 230, 3), (7, 7, 3, 64), 2],
    [(BS, 260, 260, 3), (3, 3, 3, 64), 2],
    
]


maxpoolings = [
    [(BS, 114, 114, 64), (3, 3), 2],
    [(BS, 147, 147, 64), (3, 3), 2],
    [(BS, 71, 71, 192), (3, 3), 2],
    [(BS, 167, 167, 42), (3, 3), 2],
    [(BS, 85, 85, 84), (3, 3), 2],
    [(BS, 43, 43, 336), (3, 3), 2],
    [(BS, 23, 23, 672), (3, 3), 2],
    [(BS, 113, 113, 11), (3, 3), 2],
    [(BS, 57, 57, 22), (3, 3), 2],
    [(BS, 29, 29, 88), (3, 3), 2],
]

adds = [
    [(BS, 14, 14, 1024),],
    [(BS, 28, 28, 512),],
    [(BS, 7, 7, 2048),],
    [(BS, 56, 56, 256),],
    [(BS, 21, 21, 336),],
    [(BS, 11, 11, 672),],
    [(BS, 42, 42, 168),],
    [(BS, 15, 15, 304),],
    [(BS, 14, 14, 88),],
    [(BS, 7, 7, 176),],
]

activations = [
    ['Softmax', (2048)],
    ['Softmax', (512)],
    ['Softmax', (1000)],
    ['Softmax', (100)],
    ['Softmax', (10)],
    ['Relu',  (None, 14, 14, 256)],
    ['Relu',  (None, 14, 14, 1024)],
    ['Relu',  (None, 26, 26, 512)],
    ['Relu',  (None, 5, 5, 1024)],
    ['Relu',  (None, 54, 54, 512)],
]

relu = [
    [(BS, 2048)],
    [(BS, 512)],
    [(BS, 1000)],
    [(BS, 100)],
    [(BS, 10)],
    [(BS, 57, 57, 64)],
    [(BS, 74, 74, 64)],
    [(BS, 36, 36, 192)],
    [(BS, 85, 85, 42)],
    [(BS, 43, 43, 84)],
    [(BS, 23, 23, 336)],
    [(BS, 14, 14, 672)],
    [(BS, 29, 29, 22)],
    [(BS, 14, 14, 88)],
]



resnet_convs = [
    [(1, 230, 230, 3), (7, 7, 3, 64), 2],
    [(1, 58, 58, 64), (3, 3, 64, 64), 1],
    [(1, 58, 58, 64), (3, 3, 64, 64), 1],
    [(1, 58, 58, 64), (3, 3, 64, 64), 1],
    [(1, 58, 58, 64), (3, 3, 64, 64), 1],
    [(1, 58, 58, 64), (3, 3, 64, 128), 2],
    [(1, 30, 30, 128), (3, 3, 128, 128), 1],
    [(1, 56, 56, 64), (1, 1, 64, 128), 2],
    [(1, 30, 30, 128), (3, 3, 128, 128), 1],
    [(1, 30, 30, 128), (3, 3, 128, 128), 1],
    [(1, 30, 30, 128), (3, 3, 128, 256), 2],
    [(1, 16, 16, 256), (3, 3, 256, 256), 1],
    [(1, 28, 28, 128), (1, 1, 128, 256), 2],
    [(1, 16, 16, 256), (3, 3, 256, 256), 1],
    [(1, 16, 16, 256), (3, 3, 256, 256), 1],
    [(1, 16, 16, 256), (3, 3, 256, 512), 2],
    [(1, 9, 9, 512), (3, 3, 512, 512), 1],
    [(1, 14, 14, 256), (1, 1, 256, 512), 2],
    [(1, 9, 9, 512), (3, 3, 512, 512), 1],
    [(1, 9, 9, 512), (3, 3, 512, 512), 1],
]




types = [
    # ('matmul', matmuls), 
    # ('conv_2d', conv_2ds), 
    # ('maxpooling', maxpoolings), 
    # ('add', adds),
    # ('relu', relu)
    
    ('conv_2d', resnet_convs), 
    
]


all_operations = {}

for type_name, type_opreations in types:
    
    for op in tqdm(type_opreations, desc=f'{type_name}'):
        
        maps = None
        if type_name == 'matmul':

            A, B = op
            
            if A[0] < 5000 and A[1] < 5000 and B[0] < 5000 and B[1] < 5000:
                raw_operation = f"linalg.matmul ins(%arg0, %arg1 : tensor<{A[0]}x{A[1]}xf32>, tensor<{B[0]}x{B[1]}xf32>) outs(%arg2 : tensor<{A[0]}x{B[1]}xf32>) -> tensor<{A[0]}x{B[1]}xf32>"
            else:
                continue
        
        
        elif type_name == 'conv_2d':
            
            input_shape, kernel, stride = op
            N, H, W, C = input_shape
            KH, KW, C, F = kernel

            dilation = 1
            padding = 0

            W_ = ((W + 2 * padding - dilation * (KW - 1) - 1) // stride) + 1
            H_ = ((H + 2 * padding - dilation * (KH - 1) - 1) // stride) + 1

            raw_operation = f"linalg.conv_2d_nchw_fchw {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{F}x{C}x{KH}x{KW}xf32>) outs (%init: tensor<{N}x{F}x{H_}x{W_}xf32>) -> tensor<{N}x{F}x{H_}x{W_}xf32>"


        elif type_name == 'maxpooling':
            # [(None, 114, 114, 64), (3, 3), 2],
            
            input_shape, kernel, stride = op
            N, H, W, C = input_shape
            K, _ = kernel
 
            dilation = 1

            H_ = (H - dilation * (K - 1) - 1) // stride + 1
            W_ = (W - dilation * (K - 1) - 1) // stride + 1

            raw_operation = f"linalg.pooling_nchw_max {{dilations = dense<{dilation}> : tensor<2xi64>, strides = dense<{stride}> : tensor<2xi64>}} ins (%input, %filter: tensor<{N}x{C}x{H}x{W}xf32>, tensor<{K}x{K}xf32>) outs (%init: tensor<{N}x{C}x{H_}x{W_}xf32>) -> tensor<{N}x{C}x{H_}x{W_}xf32>"


        elif type_name == 'add':
            # [(BS, 11, 11, 672),],
            input_shape, = op
            input_shape = list(map(str, input_shape))
            SHAPE = 'x'.join(input_shape)
            
            raw_operation = f"linalg.add ins(%arg0, %arg1: tensor<{SHAPE}xf32>, tensor<{SHAPE}xf32>) outs(%arg2: tensor<{SHAPE}xf32>) -> tensor<{SHAPE}xf32>"
        
        elif type_name == 'relu':
            input_shape = op[0]

            if len(input_shape) == 2:
                input_shape = list(map(str, input_shape))
                SHAPE = 'x'.join(input_shape)
                
                maps = """
                #map2 = affine_map<(d0, d1) -> (d0, d1)>
                """.strip()

                raw_operation = """
                linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<SHAPExf32>) outs(%35 : tensor<SHAPExf32>) {
                    ^bb0(%in: f32, %out: f32):
                    %cst_1 = arith.constant 0.000000e+00 : f32
                    %46 = arith.cmpf ugt, %in, %cst_1 : f32
                    %47 = arith.select %46, %in, %cst_1 : f32
                    linalg.yield %47 : f32
                } -> tensor<SHAPExf32>
                """.strip().replace('SHAPE', SHAPE)
            
            elif len(input_shape) == 4:
                input_shape = list(map(str, input_shape))
                SHAPE = 'x'.join(input_shape)
                
                maps = """
                #map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                #map2 = affine_map<(d0, d1, d2, d3) -> (0, d1, d2, d3)>
                """.strip()

                raw_operation = """
                linalg.generic {indexing_maps = [#map2, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%28 : tensor<SHAPExf32>) outs(%25 : tensor<SHAPExf32>) {
                    ^bb0(%in: f32, %out: f32):
                    %cst_1 = arith.constant 0.000000e+00 : f32
                    %90 = arith.cmpf ugt, %in, %cst_1 : f32
                    %91 = arith.select %90, %in, %cst_1 : f32
                    linalg.yield %91 : f32
                } -> tensor<SHAPExf32>
                """.strip().replace('SHAPE', SHAPE)
                
            
        
        
        print(raw_operation)
    
        wrapped_operation = function_wrapper(raw_operation, maps=maps)  
        loops = lower_linalg_to_loops(wrapped_operation, 'examples/temp_mlir.mlir')            
        
        loops_data = get_nested_loops_data(loops)
        
        transform_wrapped_operation = transform_wrapper(raw_operation, maps=maps)
        
        # continue
        exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 300, 'examples/temp_mlir.mlir')
        if exec_time and exec_time < 1000000:
            exec_time = np.median([exec_time] + [evaluate_code_with_timeout(transform_wrapped_operation, 300, 'examples/temp_mlir.mlir') for _ in range(5)])
            
        
        if exec_time:
            all_operations[f"{raw_operation}"] = {
                "operation": raw_operation,
                "wrapped_operation": wrapped_operation,
                "lowered_operation": loops,
                "transform_wrapped_operation": transform_wrapped_operation,
                "loops_data": loops_data,
                "execution_time":exec_time,
            }
        else:
            continue
        
        print(exec_time)
        
        # print(raw_operation, end='\n\n\n')
        # print(wrapped_operation, end='\n\n\n')
        # print(loops, end='\n\n\n')
        # print(transform_wrapped_operation, end='\n\n\n')
        # print(loops_data, end='\n\n\n')
            
        with open(f"./generated_data/resnet18_convs.json", "w") as file:
            json.dump(all_operations, file)