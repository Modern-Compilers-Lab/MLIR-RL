from utils.observation_utils import build_nested_loops_feature_vector
from utils.consts import PPO_ACTIONS
from fusion_utils.transforms import *
from utils.observation_utils import *
import multiprocessing

import torch

import numpy as np
import random
import json
from copy import deepcopy, copy
from dataclasses import dataclass

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



@dataclass
class OperationState:
    operation_line: str
    operation_dict: dict
    operation_id: str
    producer_id: int
    producers: list
    transformed_code: str
    transformed_all_code: str
    actions: np.array
    actions_mask: np.array
    step_count: int
    exec_time: float
    root_exec_time: float
    transformation_history: list
    cummulative_reward: float
    fused_operations: list
    operation_count: int
    operation_max: int

def get_obs(state):
    
    operation_id = state.operation_id
    if state.operation_dict[operation_id]["producers"]:
        producer_operation_id = state.operation_dict[operation_id]["producers"][state.producer_id]
    else:
        producer_operation_id = None
        
    producer_operation_id = None
    
    
    loops_data = build_nested_loops_feature_vector(state.operation_dict[state.operation_id]["loops_data"])
    
    if producer_operation_id:
        producer_loops_data = build_nested_loops_feature_vector(state.operation_dict[producer_operation_id]["loops_data"])
    else:
        producer_loops_data = np.zeros((306,), dtype=np.float64)
    action_history = state.actions.reshape(-1)
    action_mask = state.actions_mask
    
    
    obs = np.concatenate((loops_data, action_history, producer_loops_data, action_mask))
    obs[:7] = obs[:7] / 100
    obs[583:583+7] = obs[583:583+7] / 100
    
    return obs

def initialize_action_mask(action_mask):
    """
    """
    for i in range(len(PPO_ACTIONS)):
        if PPO_ACTIONS[i][0] == 'no_transformation':action_mask[i] = False
        if PPO_ACTIONS[i][0] == 'interchange':action_mask[i] = False
        if PPO_ACTIONS[i][0] == 'parallelization':action_mask[i] = True
        if PPO_ACTIONS[i][0] == 'tiling':action_mask[i] = False
        if PPO_ACTIONS[i][0] == 'fusion':action_mask[i] = False
            
    return action_mask

def update_action_mask(state, action):
    """
    """
    actions_mask = state.actions_mask
    action_name, parameters = PPO_ACTIONS[action]
    
    if "conv_2d" in state.operation_line:
        if action_name == 'parallelization':
            for i in range(len(PPO_ACTIONS)):
                if PPO_ACTIONS[i][0] == 'parallelization':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'no_transformation':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'interchange':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'tiling':actions_mask[i] = True
                if PPO_ACTIONS[i][0] == 'fusion':actions_mask[i] = False
        
        elif action_name == 'tiling':
            for i in range(len(PPO_ACTIONS)):
                if PPO_ACTIONS[i][0] == 'parallelization':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'no_transformation':actions_mask[i] = True
                if PPO_ACTIONS[i][0] == 'interchange':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'tiling':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'fusion':actions_mask[i] = True
    
    elif "matmul" in  state.operation_line:
        if action_name == 'parallelization':
            for i in range(len(PPO_ACTIONS)):
                if PPO_ACTIONS[i][0] == 'parallelization':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'no_transformation':actions_mask[i] = True
                if PPO_ACTIONS[i][0] == 'interchange':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'tiling':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'fusion':actions_mask[i] = True
        
        elif action_name == 'tiling':
            for i in range(len(PPO_ACTIONS)):
                if PPO_ACTIONS[i][0] == 'parallelization':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'no_transformation':actions_mask[i] = True
                if PPO_ACTIONS[i][0] == 'interchange':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'tiling':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'fusion':actions_mask[i] = True
    
    else:    
                
        if action_name == 'parallelization':
            for i in range(len(PPO_ACTIONS)):
                if PPO_ACTIONS[i][0] == 'parallelization':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'no_transformation':actions_mask[i] = True
                if PPO_ACTIONS[i][0] == 'interchange':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'tiling':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'fusion':actions_mask[i] = True
        
        elif action_name == 'tiling':
            for i in range(len(PPO_ACTIONS)):
                if PPO_ACTIONS[i][0] == 'parallelization':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'no_transformation':actions_mask[i] = True
                if PPO_ACTIONS[i][0] == 'interchange':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'tiling':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'fusion':actions_mask[i] = True
                
        elif action_name == 'fusion':
            for i in range(len(PPO_ACTIONS)):
                if PPO_ACTIONS[i][0] == 'parallelization':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'no_transformation':actions_mask[i] = True
                if PPO_ACTIONS[i][0] == 'interchange':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'tiling':actions_mask[i] = False
                if PPO_ACTIONS[i][0] == 'fusion':actions_mask[i] = True
    
    # print(actions_mask)
    return actions_mask
      
def apply_permutation(arr, permutation):
    return [arr[i] for i in permutation]

def sorted_divisors(n):
    divisors = []
    for i in range(2, n + 1):
        if n % i == 0:
            divisors.append(i)
    return sorted(divisors)

def get_candidates(n, num_candidates):
    div = sorted_divisors(n)
    if len(div) >= num_candidates:
        step = len(div) // num_candidates
        res = div[::step][:num_candidates]
    else:
        res = div + div[-1:]*(num_candidates-len(div))
    return res

def last_tiling(history):
    for transformation, parameters, new_exec_time, speedup in history[::-1]:
        if transformation in ['tiling', 'parallelization']:
            return parameters
    return None

def process_action(action_index, state, loops_data):

    action_name, parameter = PPO_ACTIONS[action_index]
    
    candidates = [[0]+get_candidates(upper, num_candidates=4) for (arg, lower, upper, step) in loops_data['nested_loops']]
        
    if action_name == 'interchange': # interchange
        return ['interchange', parameter]
    
    elif action_name == 'tiling': # tiling
        tiling_parameters = []
        for i, (arg, lower, upper, step) in enumerate(loops_data['nested_loops']):
            if i < len(parameter):
                if parameter[i] != -1:
                    tiling_parameters.append( candidates[i][parameter[i]])
                else: # parameter[i] == -1:
                    tiling_parameters.append( upper )
            else: # i >= len(parameter)
                tiling_parameters.append(0)
        
        last_tiling_parameters = last_tiling(state.transformation_history)
        if last_tiling_parameters is not None:
            tiling_parameters = [a if (a == 0) or ((a != 0) and (b % a == 0)) else b for a, b in zip(tiling_parameters, last_tiling_parameters)]
           
        return ['tiling', tiling_parameters]
    
    elif action_name == 'parallelization': # parallelization
        parall_parameters = []
        for i, (arg, lower, upper, step) in enumerate(loops_data['nested_loops']):
            if i < len(parameter):
                if parameter[i] != -1:
                    parall_parameters.append( candidates[i][parameter[i]])
                else: # parameter[i] == -1:
                    parall_parameters.append( upper )
            else: # i >= len(parameter)
                parall_parameters.append(0)
                
        return ['parallelization', parall_parameters]
    
    elif action_name == 'fusion': # fusion
        return ['fusion', [0]]
    
    return ['no_transformation', [0]]

def get_log_base(base):
    return lambda x: np.log(x)/np.log(base)

def speedup_reward(new, old):
    if old >= new:
        reward = old/new - 1
    else: # old < new
        reward = - new/old + 1
    reward = reward / 10
    
    # log = get_log_base(1.2)
    # reward = log(old / new)
    
    return reward


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


def build_nested_loops_feature_vector(loops_data, nested_loops_only=False):
    
    indices = [arg for (arg, lower_bound, upper_bound, step) in loops_data['nested_loops']]
    indices_dim = {arg:i for (i, arg) in enumerate(indices)}

    # Nested loop features: (upper/lower bounds, step)
    nested_loops = np.zeros((MAX_NUM_LOOPS))
    for i, (arg, lower_bound, upper_bound, step) in enumerate(loops_data['nested_loops']):
        if i == MAX_NUM_LOOPS:break
        nested_loops[i] = upper_bound

    if nested_loops_only:
        return nested_loops.reshape(-1)
    
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


def apply_transformation(state, code, transformation, parameters):
    
    print(code)
    print()
    print()
    
    
    if transformation == 'tiling':
        new_code = transform_dialect_tile(code, state.operation_id, parameters)
    elif transformation == 'parallelization':
        new_code = transform_dialect_TP(code, state.operation_id, parameters)
    elif transformation == 'interchange':
        new_code = transform_dialect_interchange(code, state.operation_id, parameters)
    elif transformation == 'vectorization':
        new_code = transform_dialect_vectorise(code, state.operation_id)
    elif transformation == 'fusion':
        consumer = state.operation_id
        producer = state.operation_dict[state.operation_id]["producers"][state.producer_id]
        print_success(f'fusion: {producer} in {consumer}')
        new_code = transform_dialect_fuse(code, consumer, producer)
    else:
        raise ValueError
    
    return new_code

def apply_transformation_wrapper(state, code, transformation, parameters, return_list):
    res = apply_transformation(state, code, transformation, parameters)
    return_list.append(res)
    
def apply_transformation_with_timeout(state, code, transformation, parameters, timeout):
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=apply_transformation_wrapper, args=(state, code, transformation, parameters, return_list))
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






class Env:
    def __init__(self, code_ast, mlir_with_tags, truncate=10, reset_repeat=1, step_repeat=1):
        self.code_ast = code_ast
        self.code = mlir_with_tags
        self.truncate = truncate
        self.get_obs = lambda state: get_obs(state)
        self.reset_repeat = reset_repeat
        self.step_repeat = step_repeat
        
        operation_dict = {}
        
        exec_times = {
            "operation_0"  : 25645684,
            "operation_1"  : 17435036960,
            "operation_2"  : 26313153,
            "operation_3"  : 3412244,
            "operation_4"  : 53482341,
            "operation_5"  : 1399209,
            "operation_6"  : 10855741983,
            "operation_7"  : 1313617,
            "operation_8"  : 189668,
            "operation_9"  : 2929765,
            "operation_10" : 2138182,
            "operation_11" : 2074,
            "operation_12" : 206884167,
            "operation_13" : 2164,
            "operation_14" : 2485,
            "operation_15" : 5540,
            "operation_16" : 1694,
            "operation_17" : 1400933,
            "operation_18" : 2265,
            "operation_19" : 2024,
            "operation_20" : 552,
            "operation_21" : 320,
            "operation_22" : 108695,
            "operation_23" : 321
        }
        
        for op in exec_times:
            exec_times[op] = 31803188473
        
        self.producers_dict = {
            'operation_2' : ['operation_1'],
            'operation_7' : ['operation_6'],
            'operation_14': ['operation_13', 'operation_12', 'operation_11'],
            'operation_19': ['operation_18', 'operation_17', 'operation_16'],
        }
        
        for op in code_ast:
            
            producers = code_ast[op]['producers']
            operation = code_ast[op]['operation']
            
            if operation.startswith('%'):
                operation = ' = '.join(operation.split(' = ')[1:])
            
            wrapped_operation = function_wrapper(operation)  
            loops = lower_linalg_to_loops(wrapped_operation)            
            
            loops_data = get_nested_loops_data(loops)
            
            transform_wrapped_operation = transform_wrapper(operation)
            
            # if op == 'operation_2':
            #     print(transform_wrapped_operation)
            #     exit()
            
            exec_time = exec_times[op]
            
            # if op in ['operation_2', 'operation_7', 'operation_14', 'operation_19']:
            if op == 'operation_2' : producers = ['operation_1']
            if op == 'operation_7' : producers = ['operation_6']
            if op == 'operation_14': producers = ['operation_13', 'operation_12', 'operation_11']
            if op == 'operation_19': producers = ['operation_18', 'operation_17', 'operation_16']

            
            operation_dict[op] = {}
            operation_dict[op]["operation"] = operation
            operation_dict[op]["producers"] = producers
            operation_dict[op]["execution_time"] = exec_time
            operation_dict[op]["loops_data"] = loops_data
            operation_dict[op]["wrapped_operation"] = wrapped_operation
            operation_dict[op]["lowered_operation"] = loops_data
            operation_dict[op]["transform_wrapped_operation"] = transform_wrapped_operation
            
            if op in ['operation_2', 'operation_7', 'operation_14', 'operation_19']:
                print(op, producers)
            # print(op, ':\n', operation_dict[op]["transform_wrapped_operation"])
        
        self.operation_dict = operation_dict
            
    def reset(self):
        
        operation_dict = self.operation_dict

        # possible_operations = ['operation_19']
        possible_operations = ['operation_19', 'operation_14', 'operation_7', 'operation_2']
        # possible_operations = [ i for i in self.operation_dict]
        # possible_operations = [ i for i in self.operation_dict if "linalg.conv_2d" in operation_dict[i]["operation"]]
        # print(possible_operations)
        # exit()
        
        self.possible_operations = possible_operations
        
        # operation_id = random.choice(possible_operations)
        operation_id = possible_operations[0]
        
        print_info("current operation:", operation_id)
        
        
        # print_success(operation_id, operation_dict[operation_id]["operation"])
 
        exec_time = operation_dict[operation_id]['execution_time']

        
        actions_mask = np.ones((len(PPO_ACTIONS),), dtype=np.bool_)
        actions_mask = initialize_action_mask(actions_mask)

        num_loops = len(operation_dict[operation_id]["loops_data"]["nested_loops"])

        state = OperationState(
            operation_line=operation_dict[operation_id]["operation"],
            operation_dict=operation_dict,
            operation_id=operation_id,
            producer_id=0,
            producers=operation_dict[operation_id]["producers"],
            # transformed_code=operation_dict[operation_id]["transform_wrapped_operation"],
            transformed_code=self.code,
            transformed_all_code=self.code,
            actions=np.zeros((len(PPO_ACTIONS),)),
            actions_mask=actions_mask,
            step_count=0,
            exec_time=exec_time,
            root_exec_time=exec_time,
            transformation_history=[],
            cummulative_reward=0,
            fused_operations=[],
            operation_count=0,
            operation_max=4,
        )

        obs = self.get_obs(state)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = torch.unsqueeze(obs, 0)
        
        return state, obs
    
    def next_operation(self, state):
        
        # print_info("inside next_operation", state.operation_count)
        
        operation_id = self.possible_operations[state.operation_count]
        
        print_info("current operation:", operation_id)
        
        exec_time = state.operation_dict[operation_id]['execution_time']
        
        actions_mask = np.ones((len(PPO_ACTIONS),), dtype=np.bool_)
        actions_mask = initialize_action_mask(actions_mask)
        
        next_state = OperationState(
            operation_line=state.operation_dict[operation_id]["operation"],
            operation_dict=state.operation_dict,
            operation_id=operation_id,
            producer_id=0,
            producers=self.producers_dict[operation_id],
            transformed_code=self.code,
            transformed_all_code=self.code,
            actions=np.zeros((len(PPO_ACTIONS),)),
            actions_mask=actions_mask,
            step_count=0,
            exec_time=exec_time,
            root_exec_time=exec_time,
            transformation_history=[],
            cummulative_reward=0,
            fused_operations=state.fused_operations,
            operation_count=state.operation_count,
            operation_max=state.operation_max
        )
        
        obs = self.get_obs(next_state)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = torch.unsqueeze(obs, 0)
        
        return next_state, obs

    def step(self, state, action_index, model):
        """
        action_index: (action:str, parameters:list[int])
        """
        
        transformation, parameters = process_action(
            action_index=action_index,
            state=state,
            loops_data=state.operation_dict[state.operation_id]["loops_data"]
        )

        # print(action_index)
        
        
        print_success(transformation, parameters)

        reward = 0
        time_out = False


        if transformation != 'no_transformation':
            transformed_code = apply_transformation_with_timeout(
                state=state,
                code=state.transformed_code,
                transformation=transformation,
                parameters=parameters,
                timeout=60,
            )
                        
            if (transformation in ['tiling', 'parallelization']) and ('conv_2d' in state.operation_line):

                second_interchange_parameters = parameters.copy()
                second_interchange_parameters[2] = 1
                second_interchange_parameters[5] = 1
                
                # print_alert(transformed_code)
                
                transformed_code = apply_transformation_with_timeout(
                    state=state,
                    code=transformed_code,
                    transformation='parallelization',
                    parameters=second_interchange_parameters,
                    timeout=60,
                )
                
                # print(transformed_code)
                # exit()
                
                state.transformation_history += [(
                    'tiling', second_interchange_parameters, 0, 0
                    )],
                
                print_success('tiling', second_interchange_parameters)
                
            new_exec_time = None
            if transformed_code:
                # new_exec_time = evaluate_code_with_timeout(
                #     code=transformed_code,
                #     timeout=20
                # )
                # new_exec_time = random.random()
                new_exec_time = state.exec_time            
            
            if new_exec_time is not None:
                # reward += np.log(state.exec_time / new_exec_time)
                # reward += speedup_reward(new_exec_time, state.exec_time)
                reward += 0
            else:
                transformed_code = state.transformed_code
                new_exec_time = state.exec_time
                reward = -5
                time_out = True
                print_error(f'EVAL ERROR: {transformation} {parameters} {state.transformation_history}')
        else: # transformation == 'no_transformation'
            transformed_code = state.transformed_code
            new_exec_time = state.exec_time
            reward += 0


        # Update state actions:
        next_state_actions = state.actions
        next_state_actions[action_index] = state.step_count
        
        # Update action mask:
        new_actions_mask = update_action_mask(state, action_index)
            
        # print(f'{str(transformation): <20} {str(parameters): <20} {new_exec_time} {state.exec_time / new_exec_time}')
        
        next_operation = False
        if transformation == 'fusion':
            consumer = state.operation_id
            producer = state.operation_dict[state.operation_id]["producers"][state.producer_id]
            fused_operations = state.fused_operations
            fused_operations.append(producer)
            fused_operations = list(set(fused_operations))
            state.fused_operations = fused_operations
            
            # transform_dialect_tile(transformed_code, producer, [2, 2])
            
            # state.producers += state.operation_dict[producer]["producers"]
            
            if (state.producer_id + 1) < len(state.producers):
                state.producer_id += 1
            else:
                next_operation = True
                state.operation_count += 1
                # print(transformed_code)
                # exit()
            
        # print_info(state.fused_operations)

        next_state = OperationState(
            operation_line=state.operation_line,
            operation_dict=state.operation_dict,
            operation_id=state.operation_id,
            producer_id=state.producer_id,
            producers=state.producers,
            transformed_code=transformed_code,
            transformed_all_code=state.transformed_code,
            actions=next_state_actions,
            actions_mask=new_actions_mask,
            step_count=state.step_count + 1,
            exec_time=new_exec_time,
            root_exec_time=state.root_exec_time,
            transformation_history=state.transformation_history + [(
                transformation, parameters, new_exec_time, state.exec_time / new_exec_time
                )],
            cummulative_reward=state.cummulative_reward,
            fused_operations=state.fused_operations,
            operation_count=state.operation_count,
            operation_max=state.operation_max
        )

        next_obs = self.get_obs(next_state)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_obs = torch.unsqueeze(next_obs, 0)

        # done = truncate = (next_state.step_count >= self.truncate) or \
        #     (transformation == 'no_transformation') or \
        #     (transformation == 'vectorization') or \
        #     (time_out)


        if transformation == 'no_transformation':
            
            next_operation = True
            next_state.operation_count += 1
            
            if 'conv_2d' in state.operation_line:
                # print_success('decompose')
                next_state.transformed_code = apply_conv2d_decomposition(next_state.transformed_code, next_state.operation_id)
                print_success('1d' in next_state.transformed_code)
            
            vect_transformed_code = apply_transformation_with_timeout(
                state=next_state,
                code=next_state.transformed_code,
                transformation='vectorization',
                parameters=[0],
                timeout=60
            )
            
            next_state.transformed_code = vect_transformed_code
            next_state.transformation_history += [('vectorization', [0], next_state.exec_time / new_exec_time)]

        next_state.cummulative_reward += reward
        
        truncate = False
        done = next_state.operation_count >= next_state.operation_max 
        
        final_state = None
        if done:
            
            code = next_state.transformed_code
            
            code = transform_dialect_prints(code, next_state.fused_operations)
            op_operation_dict = post_process_transform_dialect_prints(code)
            
            print(code)
            
            for op in next_state.fused_operations:
                operation = op_operation_dict[op]
                wrapped_operation = function_wrapper(operation)  
                loops = lower_linalg_to_loops(wrapped_operation)            
                loops_data = get_nested_loops_data(loops)
                print(op, loops_data['nested_loops'])
                
            return
            
            new_exec_time = evaluate_code_with_timeout(code=code, timeout=60)
            print(new_exec_time)
            # new_exec_time = random.random()
            
            if new_exec_time is not None:
                # reward += np.log(next_state.exec_time / new_exec_time)
                reward += speedup_reward(new_exec_time, next_state.root_exec_time)
                next_state.exec_time = new_exec_time
            else:
                reward = -20
                time_out = True
                print_error(f'EVAL ERROR:{transformation} {parameters} {next_state.transformation_history}')
                new_exec_time = next_state.exec_time
            
            
            final_state = next_state
            next_state, next_obs = self.reset()
            
        elif next_operation:
            next_state, next_obs = self.next_operation(next_state)

        # print_success(done, reward)
        return next_obs, reward, done, truncate, next_state, final_state



class ParallelEnv:
    def __init__(self, mlir_code, num_env, truncate=10, reset_repeat=1, step_repeat=1):
        self.num_env = num_env
        self.mlir_code = mlir_code
        self.truncate = truncate
        
        with open(mlir_code, "r") as file:
            mlir_code = file.read()
            
        raw_ast_info = get_raw_ast_info(mlir_code)
        code_ast, mlir_with_tags = get_ast(raw_ast_info)


        self.env = Env(
            code_ast=code_ast,
            mlir_with_tags=mlir_with_tags,
            truncate=self.truncate,
            reset_repeat=reset_repeat,
            step_repeat=step_repeat
        )

    def reset(self):
        states, observations = [], []
        for _ in range(self.num_env):
            state, obs = self.env.reset()
            states.append(state)
            observations.append(obs)
        return states, observations

    def step(self, states, actions, model):
        batch_next_obs, batch_reward, batch_done, batch_truncate, batch_next_state = [], [], [], [], []
        batch_final_state = []
        for state, action in zip(states, actions):
            # print(state.operation)
            next_obs, reward, done, truncate, next_state, final_state = self.env.step(state, action, model)
            batch_next_obs.append( next_obs )
            batch_reward.append( reward )
            batch_done.append( done )
            batch_truncate.append( truncate )
            batch_next_state.append( next_state )
            batch_final_state.append( final_state )

        return batch_next_obs, batch_reward, batch_done, batch_truncate, batch_next_state, batch_final_state

