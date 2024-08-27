from utils.observation_utils import function_wrapper, lower_linalg_to_loops
from utils.transform_utils import evaluate_code_with_timeout
from random import randint, choice, shuffle
from tqdm import tqdm
import json, re, multiprocessing, os
from copy import copy

BATCH_SIZES = [128, 256]
SIZES = [768, 1024, 256, 1536, 2048, 512, 128, 3072]
HEIGHTS = [14, 28, 56, 7, 112, 15, 120, 150, 130, 240, 224, 228]
CHANNELS = [32, 256, 128, 192, 512, 64, 96, 48, 288, 240, 384]
KERNELS = [1, 3, 7]
DILATIONS = [1]
STRIDES = [1, 2]
SMALL, MEDIUM, BIG = 250, 500, 1000



def relu():
    
    SHAPE = "x".join([str(choice(HEIGHTS)) for _ in range(2)])
    
    relu_maps = """
    #map2 = affine_map<(d0, d1) -> (d0, d1)>
    """.strip()

    relu_operation = """
    linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel"]} ins(%38 : tensor<SHAPExf32>) outs(%35 : tensor<SHAPExf32>) {
        ^bb0(%in: f32, %out: f32):
        %cst_1 = arith.constant 0.000000e+00 : f32
        %46 = arith.cmpf ugt, %in, %cst_1 : f32
        %47 = arith.select %46, %in, %cst_1 : f32
        linalg.yield %47 : f32
        } -> tensor<SHAPExf32>
    """.strip().replace('SHAPE', SHAPE)
    
    return relu_operation, relu_maps


all_operations = {}
for i in tqdm(range(10)):
            
    raw_operation, maps = relu()
    
    # print(raw_operation)
    # if 'matmul' in operation_name:
    #     raw_operation = "linalg.matmul ins(%arg0, %arg1 : tensor<1200x1500xf32>, tensor<1500x1000xf32>) outs(%arg2 : tensor<1200x1000xf32>) -> tensor<1200x1000xf32>"
    # elif 'conv' in operation_name:
    #     # raw_operation = "linalg.conv_2d_nhwc_hwcf {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins (%input, %filter: tensor<32x230x230x3xf32>, tensor<7x7x3x64xf32>) outs (%init: tensor<32x112x112x64xf32>) -> tensor<32x112x112x64xf32>"
    #     raw_operation = "linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<2> : vector<2xi64>} ins (%input, %filter: tensor<32x3x230x230xf32>, tensor<64x3x7x7xf32>) outs (%init: tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xf32>"
    
    wrapped_operation = function_wrapper(raw_operation, maps=maps)  
    loops = lower_linalg_to_loops(wrapped_operation, 'examples/temp_mlir.mlir')
    # loops_data = get_nested_loops_data(loops)
    
    # transform_wrapped_operation = transform_wrapper(raw_operation)
    
    # # continue
    # exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 300, 'examples/temp_mlir.mlir')
    
    # if exec_time:
    #     all_operations[f"{raw_operation}"] = {
    #         "operation": raw_operation,
    #         "wrapped_operation": wrapped_operation,
    #         "lowered_operation": loops,
    #         "transform_wrapped_operation": transform_wrapped_operation,
    #         "loops_data": loops_data,
    #         "execution_time":exec_time,
    #     }
    # else:
    #     continue
    
                
    # print(raw_operation, end='\n\n\n')
    # print(wrapped_operation, end='\n\n\n')
    # print(loops, end='\n\n\n')
    # print(transform_wrapped_operation, end='\n\n\n')
    # print(loops_data, end='\n\n\n')
        
    # with open(f"./generated_data/relu.json", "w") as file:
    #     json.dump(all_operations, file)




