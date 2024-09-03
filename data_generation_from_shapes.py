from utils.observation_utils import (
    function_wrapper, 
    lower_linalg_to_loops,
    get_nested_loops_data,
    transform_wrapper,
)
from utils.transforms import evaluate_code_with_timeout
import json, re
from tqdm import tqdm
from copy import copy
import numpy as np


tmp_file = 'tmp_files/temp_mlir.mlir'


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
        loops = lower_linalg_to_loops(wrapped_operation, tmp_file)            
        
        loops_data = get_nested_loops_data(loops)
        
        transform_wrapped_operation = transform_wrapper(raw_operation, maps=maps)
        
        # continue
        exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file)
        if exec_time and exec_time < 1000000:
            exec_time = np.median([exec_time] + [evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file) for _ in range(5)])
            
        
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