from utils.observation_utils import (
    function_wrapper,
    lower_linalg_to_loops,
    formula_str_to_list,
    transform_wrapper,
    get_nested_loops_data
)
from utils.transforms import (
    transform_dialect_interchange,
    transform_dialect_TP,
    transform_dialect_tile,
    transform_dialect_vectorise,
    transform_dialect_vectorise_img2col,
    apply_conv2d_decomposition,
    transform_dialect_img2col,
    evaluate_code_with_timeout,
    get_raw_ast_info,
    get_ast,
)

import numpy as np 
from copy import copy
import os, multiprocessing, random, json

tmp_file = 'tmp_files/temp_mlir.mlir'

####################################################################################################################################


def apply_transformation(operation_tag, operation_type, code, transformation, parameters):
        
    code = code.strip()
    code = code.replace("module {\n", "")
    
    if transformation == 'T':
        new_code = transform_dialect_tile(code, operation_tag, parameters, tmp_file)
    elif transformation == 'TP':
        new_code = transform_dialect_TP(code, operation_tag, parameters, tmp_file)
    elif transformation == 'I':
        new_code = transform_dialect_interchange(code, operation_tag, parameters, tmp_file)
    elif transformation == 'img2col':
        new_code = transform_dialect_img2col(code, operation_tag, tmp_file)
    elif transformation == 'V':
        if operation_type == 'conv_2d+img2col':
            new_code = transform_dialect_vectorise_img2col(code, operation_tag, tmp_file)
        else:
            new_code = transform_dialect_vectorise(code, operation_tag, tmp_file)
    else:
        raise ValueError

    return new_code

def apply_transformation_wrapper(operation_tag, operation_type, code, transformation, parameters, return_list):
    res = apply_transformation(operation_tag, operation_type, code, transformation, parameters)
    return_list.append(res)
    
def apply_transformation_with_timeout(operation_tag, operation_type, code, transformation, parameters, timeout):
    
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=apply_transformation_wrapper, args=(operation_tag, operation_type, code, transformation, parameters, return_list))
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


####################################################################################################################################


def create_ast_TIRAMISU(current_op, expression_list):
    
    end_dict = {
        'expr_type':'access',
        'children':[]
    }
    
    operation, op1, op2 = expression_list[current_op]
    op1_ast = create_ast_TIRAMISU(op1, expression_list) if op1 != 'access' else end_dict
    op2_ast = create_ast_TIRAMISU(op2, expression_list) if op2 != 'access' else end_dict
    
    return {
        'expr_type':operation,
        'children':[
            op2_ast,
            op1_ast,
        ]
    }


def get_nested_loops_data_TIRAMISU(loops):
    
    
    lines = loops.split('\n')

    loops_detailed = {}
    loops_detailed["nested_loops"] = []
    loops_detailed["op_count"] = {'+':0, '-':0, '*':0, '/':0, 'exp':0,}
    loops_detailed["load_data"] = []
    loops_detailed["store_data"] = []

    maps = {}
    args_of_loops = []
    args_of_map = {}
    access_args = []
    ast_dict = {}

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
            
        elif "affine.store" in line:
            
            res, *alloc = line.strip().split(' ')[1:-2]
            res = res[:-1]
            alloc = ' '.join(alloc)
            args = alloc.split('[')[1][:-1].split(', ')

            for i in range(len(args)):
                if args[i] in args_of_map:
                    args[i] = args_of_map[args[i]]
                    
            loops_detailed["store_data"].append(args)
        
        elif "arith.addf" in line:loops_detailed["op_count"]['+'] += 1
        elif "arith.mulf" in line:loops_detailed["op_count"]['*'] += 1
        elif "arith.subf" in line:loops_detailed["op_count"]['-'] += 1
        elif "arith.divf" in line:loops_detailed["op_count"]['/'] += 1
        elif "math.exp" in line:loops_detailed["op_count"]['exp'] += 1
        
        if "affine.load" in line:
            arg = line.strip().split(' ')[0]
            access_args.append(arg)
        
        if any([s for s in ["arith.addf", "arith.mulf", "arith.subf", "arith.divf", "arith.maximumf"] if s in line]):
            
            res, _, _, op1, op2, _, _ = line.strip().split(' ')
            op1, op2 = op1[:-1], op2
            op1 = 'access' if op1 in access_args else op1
            op2 = 'access' if op2 in access_args else op2
            
            if   "arith.addf" in line:
                ast_dict[res] = ['add', op1, op2]
            elif "arith.mulf" in line:
                ast_dict[res] = ['mul', op1, op2]
            elif "arith.subf" in line:
                ast_dict[res] = ['sub', op1, op2]
            elif "arith.divf" in line:
                ast_dict[res] = ['div', op1, op2]
            elif "arith.maximumf" in line:
                ast_dict[res] = ['max', op1, op2]
            
    
    # print(maps)
    # print(args_of_map)
    # print(loops_detailed)
    
    final_arg = list(ast_dict.keys())[-1]
    ast_structure = create_ast_TIRAMISU(final_arg, ast_dict)
    
    return loops_detailed, ast_structure


def get_annotations_TIRAMISU(loops_data):
    
    indices = [arg for (arg, lower_bound, upper_bound, step) in loops_data['nested_loops']]
    indices_dim = {arg:i for (i, arg) in enumerate(indices)}

    # Nested loop features: (upper/lower bounds, step)
    iterators = {}
    for i in range(len(loops_data['nested_loops'])):
        arg, lower_bound, upper_bound, step = loops_data['nested_loops'][i]
        iterators[arg] = {
            "lower_bound": str(lower_bound),
            "upper_bound": str(upper_bound),
            "parent_iterator": None if i == 0 else loops_data['nested_loops'][i-1][0],
            "child_iterators": [ loops_data['nested_loops'][i+1][0] ] if i != len(loops_data['nested_loops'])-1 else [],
            "computations_list": [ "comp00" ] if i == len(loops_data['nested_loops'])-1 else []
        },
        
        
    computations = {"comp00":{}}
    computations["comp00"] = {
        "absolute_order": 0,
        "iterators": indices,
        "comp_is_reduction": False,
        "number_of_additions": loops_data["op_count"]["+"],
        "number_of_subtraction": loops_data["op_count"]["-"],
        "number_of_multiplication": loops_data["op_count"]["*"],
        "number_of_division": loops_data["op_count"]["/"],
    }
    
    # load access matrices:
    
    load_data = loops_data["load_data"]
    accesses = []

    for load_i, load in enumerate(load_data):
        dimensions_terms = [formula_str_to_list(term) for term in load]
        mat = np.zeros((len(dimensions_terms), len(indices_dim)), dtype=np.int64)
        for m, dimension_term in enumerate(dimensions_terms):
            for index, factor in dimension_term:
                if index in indices_dim:
                    n = indices_dim[index]
                    mat[m, n] = factor        
        accesses.append({
            "access_is_reduction": False,
            "buffer_id": load_i,
            "access_matrix":mat.tolist()
        })

    computations["comp00"]["accesses"] = accesses

    annotations = {
        "program_annotation":{
            "iterators": iterators,
            "computations": computations,
            "initial_execution_time": None,
        }
    }
    
    return annotations

                
def get_tiramisu_representation_TIRAMISU(raw_operation, tmp_file, evaluate=False):
    wrapped_operation = function_wrapper(raw_operation)
    loops = lower_linalg_to_loops(wrapped_operation, tmp_file)            
    loops_data, ast = get_nested_loops_data_TIRAMISU(loops)
    annotations = get_annotations_TIRAMISU(loops_data)
    
    annotations["filename"] = raw_operation
    annotations["raw_operation"] = raw_operation
    annotations["node_name"] = os.uname()[1]
    annotations["program_annotation"]["computations"]["comp00"]["expression_representation"] = ast
    
    
    if evaluate:
        transform_wrapped_operation = transform_wrapper(raw_operation)
        # Evaluate the execution time of the transformed operation with a timeout of 300 seconds
        exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file)
        # If the execution time is valid and below a certain threshold, calculate a more stable median execution time
        if exec_time and exec_time < 1000000:
            exec_time = np.median([exec_time] + [evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file) for _ in range(2)])
        
        annotations["initial_execution_time"] = exec_time
        
    return annotations


####################################################################################################################################




# import json
# print(json.dumps(annotations, indent=4))


####################################################################################################################################


def get_schedules_list_TIRAMISU(raw_operation, random_schedules, tmp_file):

    random_schedules = [
        [('TP', [2,]), ('T', [2,]), ],
        [('TP', [4,]), ('T', [4,]), ],
        [('TP', [8,]), ('T', [8,]), ],
    ]


    transform_wrapped_operation = transform_wrapper(raw_operation)

    raw_ast_info = get_raw_ast_info(transform_wrapped_operation, tmp_file)
    code_ast, code_with_tags = get_ast(raw_ast_info)
    transform_wrapped_operation = code_with_tags
    operation_tag = list(code_ast.keys())[-1]


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



    schedules_list = []

    for schedule in random_schedules:
        
        current_state = transform_wrapped_operation
        sched_str = ""
        
        for trans, param in schedule:
                            
            # Decompose convolution:
            if (trans == 'V' and operation_type == 'conv_2d'):
                tile_param = [0]*6
                if ('conv_2d_nhwc_hwcf' in raw_operation):      tile_param = [0, 1, 0, 0, 1, 0, 0] 
                elif ('conv_2d_nchw_fchw' in raw_operation):    tile_param = [0, 0, 1, 0, 0, 1, 0]
                elif ('pooling' in raw_operation):              tile_param = [0, 0, 1, 0, 1, 0, 0]   
                current_state = apply_transformation_with_timeout(
                    operation_tag=operation_tag,
                    operation_type=operation_type,
                    code=current_state,
                    transformation='T',
                    parameters=tile_param,
                    timeout=20,
                )
                current_state = apply_conv2d_decomposition(current_state, operation_tag, tmp_file)

            
            new_state = apply_transformation_with_timeout(
                    operation_tag=operation_tag,
                    operation_type=operation_type,
                    code=current_state,
                    transformation=trans,
                    parameters=param,
                    timeout=20,
            )
            
            
            # exec_time = evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file)
            # if exec_time and exec_time < 1000000:
            #     exec_time = [exec_time] + [evaluate_code_with_timeout(transform_wrapped_operation, 300, tmp_file) for _ in range(2)]
            # else:
            #     exec_time = [exec_time]
            exec_time = [round(random.random(), 7) for _  in range(10)]
            
            sched_str = sched_str + f'{trans}({param})'      
            info_dict = {
                "comp00": {
                    "parallelization+tiling":param if trans == 'TP' else None,
                    "tiling": param if trans == 'T' else None,
                    "interchange": param if trans == 'I' else None,
                    "vectorization": True if trans == 'V' else False,
                },
                "execution_times": exec_time,
                "sched_str": sched_str
            }
            
            
            current_state = new_state
            
            schedules_list.append(info_dict)
    
    return schedules_list

raw_operations = [
    "linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins (%input, %filter: tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) outs (%init: tensor<256x256x12x12xf32>) -> tensor<256x256x12x12xf32>",
    "linalg.add ins(%arg0, %arg1: tensor<28x150x7x130xf32>, tensor<28x150x7x130xf32>) outs(%arg2: tensor<28x150x7x130xf32>) -> tensor<28x150x7x130xf32>",
    "linalg.matmul ins(%arg0, %arg1 : tensor<256x128xf32>, tensor<128x256xf32>) outs(%arg2 : tensor<256x256xf32>) -> tensor<256x256xf32>",
    "linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins (%input, %filter: tensor<128x288x224x150xf32>, tensor<3x3xf32>) outs (%init: tensor<128x288x111x74xf32>) -> tensor<128x288x111x74xf32>",
]

all_annotations = {}

for raw_operation in raw_operations:
    annotations = get_tiramisu_representation_TIRAMISU(raw_operation, tmp_file)
    schedules_list = get_schedules_list_TIRAMISU(raw_operation, [], tmp_file)
    annotations['schedules_list'] = schedules_list
    all_annotations[raw_operation] = annotations

# print(json.dumps(all_annotations, indent=4))

with open('./generated_data/tiramisu_annotations.json', "w") as file:
    json.dump(all_annotations, file)