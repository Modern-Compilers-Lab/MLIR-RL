from utils.observation_utils import build_nested_loops_feature_vector
from fusion_utils.transforms import (
    transform_dialect_tile,
    transform_dialect_TP,
    transform_dialect_interchange,
    transform_dialect_vectorise,
    apply_conv2d_decomposition,
    get_raw_ast_info,
    get_ast
)

from utils.consts import (
    MAX_NUM_STORES_LOADS,
    MAX_NUM_LOOPS,
    MAX_NUM_LOAD_STORE_DIM,
    PPO_ACTIONS,
    INTERCHANGE_ACTIONS
)

import os
import torch

import numpy as np
import random
import json
from copy import deepcopy
from dataclasses import dataclass
import multiprocessing
from tqdm import tqdm



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




def apply_transformation(state, code, transformation, parameters):
    
    # print(transformation, parameters)
    
    code = code.strip()
    code = code.replace("module {\n", "")
    # code = code.replace("\n}", "")
    # print('\n\n\n')
    
    # print(code)
    # print('\n\n\n')
        
    if transformation == 'tiling':
        new_code = transform_dialect_tile(code, state.operation_tag, parameters)
    elif transformation == 'parallelization':
        new_code = transform_dialect_TP(code, state.operation_tag, parameters)
    elif transformation == 'interchange':
        new_code = transform_dialect_interchange(code, state.operation_tag, parameters)
    elif transformation == 'vectorization':
        new_code = transform_dialect_vectorise(code, state.operation_tag)
    else:
        raise ValueError
    
    # print(new_code)
    # exit()
    
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




def evaluate_code(code, timeout=20):
    # command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_1 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -scf-foreach-thread-lowering -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts"""
    command_2 = """/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-cpu-runner -e main -entry-point-result=void -shared-libs=/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libmlir_c_runner_utils.so,/scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/lib/libomp.so"""
    
    tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/examples/temp_mlir.mlir"
    # tmp_file = "generated_mlir/bigger_input_nn.mlir"
    
    os.environ["OMP_NUM_THREADS"] = "8"
    
    with open(tmp_file, "w") as file:
        file.write(code)
        
    out = os.popen(f"""{command_1} {tmp_file} | {command_2} /dev/stdin""").read()
    # out = os.popen(f"""{command_1} {tmp_file}""").read()
    
    if out:
        return int(out.strip().split('\n')[-1])
    else:
        return None

def evaluate_code_wrapper(code, return_list, state):
    res = evaluate_code(code)
    # if res:res /= 21
    # if state:
        # if 'conv' in state.operation_id:res /= 54
    return_list.append(res)

def evaluate_code_with_timeout(code, timeout, state=None):
    manager = multiprocessing.Manager()
    return_list = manager.list()
    process = multiprocessing.Process(target=evaluate_code_wrapper, args=(code, return_list, state))
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





@dataclass
class OperationState:
    operation_tag: str
    operation_file: str
    operation: str
    operation_id: str
    wrapped_operation: str
    lowered_operation: str
    loops_data: dict
    transformed_code: str
    actions: np.array
    actions_mask: np.array
    step_count: int
    exec_time: float
    root_exec_time: float
    interchange_history: list
    transformation_history: list
    cummulative_reward: float

def get_obs(state):
    
    loops_data = build_nested_loops_feature_vector(state.loops_data)
    
    
    action_history = state.actions.reshape(-1)
    action_mask = state.actions_mask
    
    
    
    # print(loops_data.shape, action_history.shape, action_mask.shape)
    obs = np.concatenate((loops_data, action_history, action_mask))
    obs[:7] = obs[:7] / 100
    return obs

def initialize_action_mask(action_mask, num_loops):
    """
    Action mask (4 + L + L + (L-1) + (L-2) + (L-3) ):
        Transformations: end, TP, T, Interchange
        TP: L loops
        T : L loops
        Interchange: 2-consecutive interchanges: L - 1
                   : 3-consecutive interchanges: L - 2
                   : 4-consecutive interchanges: L - 3
        Interchange: 3L - 6
        
    action_mask[:4] = [end, TP, T, I]
    """
    L = MAX_NUM_LOOPS
    
    TP_BEGIN = 4
    T_BEGIN = TP_BEGIN + L
    I_BEGIN_2C = T_BEGIN + L
    I_BEGIN_3C = I_BEGIN_2C + (L-1)
    I_BEGIN_4C = I_BEGIN_3C + (L-2)
    
    action_mask[:4] = [False, True, False, False]
    action_mask[TP_BEGIN+num_loops:T_BEGIN] = False
    action_mask[T_BEGIN+num_loops:I_BEGIN_2C] = False
    action_mask[I_BEGIN_2C+num_loops-1:I_BEGIN_3C] = False
    action_mask[I_BEGIN_3C+num_loops-2:I_BEGIN_4C] = False
    action_mask[I_BEGIN_4C+num_loops-3:] = False
    
    if num_loops == 1:
        action_mask[3] = False
        action_mask[I_BEGIN_2C] = True
    
    return action_mask

def update_action_mask(state, transformation, parameters, num_loops):
    """
    actions_mask: (4 + L + L + (L-1) + (L-2) + (L-3) )
    action_mask[:4] = [end, TP, T, I]
    """
    
    L = MAX_NUM_LOOPS
    
    TP_BEGIN = 4
    T_BEGIN = TP_BEGIN + L
    I_BEGIN_2C = T_BEGIN + L
    I_BEGIN_3C = I_BEGIN_2C + (L-1)
    I_BEGIN_4C = I_BEGIN_3C + (L-2)
    
    actions_mask = state.actions_mask
    
    if "pooling" in state.operation_id or "conv_2d" in state.operation_id:
        if transformation == 'parallelization':actions_mask[:4] = [False, False, True, False]
        if transformation == 'tiling':actions_mask[:4] = [True, False, False, False]
    
    elif "matmul" in  state.operation_id:
        if transformation == 'parallelization':actions_mask[:4] = [True, False, True, True]
        if transformation == 'tiling':actions_mask[:4] = [True, False, True, True]
        
    else:
        if transformation == 'parallelization':actions_mask[:4] = [True, False, True, False]
        # if transformation == 'interchange':actions_mask[:4] = [True, False, True, True]
        if transformation == 'tiling':actions_mask[:4] = [True, False, True, False]
    
    if num_loops == 1:
        actions_mask[3] = False
        actions_mask[I_BEGIN_2C] = True
    
    return actions_mask

def update_action_history(state: OperationState, transformation, parameters):
    # actions.shape: (L, 3, truncate)
    # parallelization, tiling, interchange
     
    num_loops = len(state.loops_data["nested_loops"])
    actions = state.actions
    
    for l in range(num_loops):
    
        if transformation == 'parallelization':
            actions[l, 0, state.step_count] = parameters[l]
        elif transformation == 'tiling':
            actions[l, 1, state.step_count] = parameters[l]
        elif transformation == 'interchange':
            actions[l, 2, state.step_count] = parameters[l]
    
    return actions
    
    

def apply_permutation(arr, permutation):
    return [arr[i] for i in permutation]

def sorted_divisors(n):
    divisors = []
    for i in range(2, n + 1):
        if n % i == 0:
            divisors.append(i)
    return sorted(divisors)

def get_candidates(n, num_candidates):
    
    if n == 1:
        return [1]*num_candidates
    
    
    div = sorted_divisors(n)
    if len(div) >= num_candidates:
        step = len(div) // num_candidates
        res = div[::step][:num_candidates]
    else:
        res = div + div[-1:]*(num_candidates-len(div))
    return res

def last_tiling(history):
    for transformation, parameters in history[::-1]:
        if transformation in ['tiling', 'parallelization']:
            return parameters
    return None


def clean_action(state, transformation, parameters):
    
    
    if 'conv' in state.operation_id:
        if transformation == 'parallelization':
            parameters = [get_candidates(upper, num_candidates=4)[1] if i < 3 else 0 for i, (arg, lower, upper, step) in enumerate(state.loops_data['nested_loops'])]
    
    if transformation == 'parallelization':
        if parameters.count(0) == len(parameters):
            parameters = [get_candidates(upper, num_candidates=4)[1] if i < 2 else 0 for i, (arg, lower, upper, step) in enumerate(state.loops_data['nested_loops'])]
    
    if transformation == 'tiling':
        transformations = [a for (a,b) in state.transformation_history]
        if transformations.count('tiling') >= 1:
            transformation =  None

    if transformation == 'interchange':
        transformations = [a for (a,b) in state.transformation_history]
        if transformations.count('interchange') >= 1:
            transformation =  None
            
    return transformation, parameters
    

def process_action(action_index, state: OperationState):

    loops_data = state.loops_data
    num_loop = len(loops_data['nested_loops'])
    action_name, parameter = action_index

    candidates = [[0]+get_candidates(upper, num_candidates=4) for (arg, lower, upper, step) in loops_data['nested_loops']]

    if action_name == 'interchange': # interchange
        parameters = INTERCHANGE_ACTIONS[parameter]
        parameters = parameters[:num_loop]
        return ['interchange', list(parameters)]
    
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
        
        # if 'conv_2d' in state.operation_id:
        #     parall_parameters = [n if n != 0 else 1 for n in parall_parameters]
        return ['parallelization', parall_parameters]
    
    return ['no_transformation', [0]]

def get_log_base(base):
    return lambda x: np.log(x)/np.log(base)

def speedup_reward(new, old):
    if old >= new:
        reward = old/new - 1
    else: # old < new
        reward = - new/old + 1
    # reward = reward / 1000
    
    # log = get_log_base(1.2)
    # reward = log(old / new)
    
    return reward


class Env:
    def __init__(self, operations_files, truncate=10, reset_repeat=1, step_repeat=1):
        
        # operations_files = [file for file in operations_files if any([ s in file[0] for s in ['matmul', 'conv', 'generic', 'pool'] ]) ]
        
        operations_files = [[details['operation']]+[details] for file, details in operations_files.items()]

        # operations_files = [file for file in operations_files if any([ s in file[0] for s in ['matmul'] ]) ]
        
        self.operations_files = operations_files
        self.truncate = truncate
        self.get_obs = lambda state: get_obs(state)
        self.reset_repeat = reset_repeat
        self.step_repeat = step_repeat
        
        
        for i in tqdm(range(len(operations_files))):
            code = operations_files[i][1]["transform_wrapped_operation"]
            raw_ast_info = get_raw_ast_info(code)
            code_ast, code_with_tags = get_ast(raw_ast_info)
            operations_files[i][1]["transform_wrapped_operation"] = code_with_tags
            
            operation_tag = list(code_ast.keys())[-1]
            operations_files[i][1]["operation_tag"] = operation_tag
            
            
    def reset(self, idx=None):
        operations_files = self.operations_files
        # operations_files = [file for file in self.operations_files if any([ s in file[0] for s in ['conv_2d'] ]) ]
        # operations_files = operations_files[:50]
        # print(len(operations_files))
        # exit()
        
        if idx:
            operation_file, operation_dict = operations_files[idx]
        else:
            operation_file, operation_dict = random.choice(operations_files)
            
        num_loops = len(operation_dict["loops_data"]["nested_loops"])
        
        operation_id = operation_file
 

        # exec_time = evaluate_code_2(
        #     code=operation_dict["transform_wrapped_operation"],
        # )
        
        exec_time = operation_dict['execution_time']

        # Action mask:
        # Transformations: TP, T, Interchange
        # TP: L loops
        # T : L loops
        # Interchange: 2-consecutive interchanges: L - 1
        #            : 3-consecutive interchanges: L - 2
        #            : 4-consecutive interchanges: L - 3
        # Interchange: 3L - 6
        actions_mask = np.ones((4 + MAX_NUM_LOOPS + MAX_NUM_LOOPS + 3*MAX_NUM_LOOPS - 6), dtype=np.bool_)
        actions_mask = initialize_action_mask(actions_mask, num_loops)

        # Action history:
        # 3 because we have 3 transformations: TP, T, I
        actions = np.zeros((MAX_NUM_LOOPS, 3, self.truncate,))
        

        state = OperationState(
            operation_tag=operation_dict["operation_tag"],
            operation_file=operation_file,
            operation=operation_dict["operation"],
            operation_id=operation_id,
            wrapped_operation=operation_dict["wrapped_operation"],
            lowered_operation=operation_dict["lowered_operation"],
            loops_data=operation_dict["loops_data"],
            transformed_code=operation_dict["transform_wrapped_operation"],
            # actions=np.zeros((self.truncate, len(PPO_ACTIONS))),
            actions=actions,
            actions_mask=actions_mask,
            step_count=0,
            exec_time=exec_time,
            root_exec_time=exec_time,
            interchange_history=[list(range(num_loops))],
            transformation_history=[],
            cummulative_reward=0
        )

        obs = self.get_obs(state)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = torch.unsqueeze(obs, 0)
        
        return state, obs
        

    def step(self, state, action_index, model):
        """
        action_index: (action:str, parameters:list[int])
        """
                
        num_loops = len(state.loops_data["nested_loops"])

        transformation, parameters = process_action(
            action_index=action_index,
            state=state
        )

        print_success(transformation, parameters)
        # ptrans, pparam = clean_action(state, transformation, parameters)
        # if ptrans:
            # print_success(ptrans, pparam)

        reward = 0
        time_out = False

        # print(state.transformed_code)
        
        if transformation != 'no_transformation':
            
            transformed_code = apply_transformation_with_timeout(
                state=state,
                code=state.transformed_code,
                transformation=transformation,
                parameters=parameters,
                timeout=20,
            )
            
            if (transformation == 'tiling'):
                
                if ('conv_2d' in state.operation_id or 'pooling' in state.operation_id):

                    if ('conv_2d_nhwc_hwcf' in state.operation_id):
                        second_interchange_parameters = parameters.copy()
                        second_interchange_parameters[1] = 1
                        second_interchange_parameters[4] = 1
                        
                    elif ('conv_2d_nchw_fchw' in state.operation_id):
                        second_interchange_parameters = parameters.copy()
                        second_interchange_parameters[2] = 1
                        second_interchange_parameters[5] = 1
                        
                    elif ('pooling' in state.operation_id):
                        # second_interchange_parameters = parameters.copy()
                        second_interchange_parameters = [0]*len(parameters)
                        second_interchange_parameters[2] = 1
                        second_interchange_parameters[4] = 1
                        
                    transformed_code = apply_transformation_with_timeout(
                        state=state,
                        code=transformed_code,
                        transformation=transformation,
                        parameters=second_interchange_parameters,
                        timeout=20,
                    )
                    print_success(transformation, second_interchange_parameters)
                
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
        next_state_actions = update_action_history(state, transformation, parameters)
        
        # Update action mask:
        new_actions_mask = update_action_mask(state, transformation, parameters, num_loops)
            
        # print(f'{str(transformation): <20} {str(parameters): <20} {new_exec_time} {state.exec_time / new_exec_time}')
        


        next_state = OperationState(
            operation_tag=state.operation_tag,
            operation_file=state.operation_file,
            operation=state.operation,
            operation_id=state.operation_id,
            wrapped_operation=state.wrapped_operation,
            lowered_operation=state.lowered_operation,
            loops_data=state.loops_data,
            transformed_code=transformed_code,
            actions=next_state_actions,
            actions_mask=new_actions_mask,
            step_count=state.step_count + 1,
            exec_time=new_exec_time,
            root_exec_time=state.root_exec_time,
            interchange_history=deepcopy(state.interchange_history),
            transformation_history=state.transformation_history + [(transformation, parameters)],
            cummulative_reward=state.cummulative_reward
        )

        next_obs = self.get_obs(next_state)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_obs = torch.unsqueeze(next_obs, 0)

        done = truncate = (next_state.step_count >= self.truncate) or \
            (transformation == 'no_transformation') or \
            (transformation == 'vectorization') or \
            (time_out)


        if done:
            
            if 'conv_2d' in state.operation_id:
                next_state.transformed_code = apply_conv2d_decomposition(next_state.transformed_code, next_state.operation_tag)
                # if 'conv_2d_nchw_fchw' in state.operation_id:
                #     next_state.transformed_code = apply_conv_2d_nchw_fchw_decomposition(next_state.transformed_code)
                # elif 'conv_2d_nhwc_hwcf' in state.operation_id:
                #     next_state.transformed_code = apply_conv_2d_nhwc_hwcf_decomposition(next_state.transformed_code)           
                
                print_success('1d' in next_state.transformed_code)
                
            elif 'pooling' in state.operation_id:
                next_state.transformed_code = apply_conv2d_decomposition(next_state.transformed_code, next_state.operation_tag)
                # next_state.transformed_code = apply_maxpool_decomposition(next_state.transformed_code)
                print_success('linalg.pooling_ncw_max' in next_state.transformed_code)
                
            vect_transformed_code = apply_transformation_with_timeout(
                state=next_state,
                code=next_state.transformed_code,
                transformation='vectorization',
                parameters=[0],
                timeout=20
            )
            
            
            new_exec_time = None
            if vect_transformed_code:
                # print_info('vector.' in vect_transformed_code)
                
                new_exec_time = evaluate_code_with_timeout(code=vect_transformed_code, timeout=30, state=next_state)
                # new_exec_time = random.random()
            if new_exec_time is not None:
                r = speedup_reward(new_exec_time, next_state.root_exec_time)
                
                # clip speedup for non matmul operations:
                if not "matmul" in next_state.operation:
                    r = min(r, 500)
                
                reward += r
                next_state.transformed_code = vect_transformed_code
                next_state.exec_time = new_exec_time
            else:
                reward = -20
                time_out = True
                print_error(f'EVAL ERROR:{transformation} {parameters} {next_state.transformation_history}')
                new_exec_time = next_state.exec_time
            
            next_state.transformation_history += [('vectorization', [0])]

        next_state.cummulative_reward += reward

        final_state = None
        if done or truncate:
            final_state = next_state
            next_state, next_obs = self.reset()

        return next_obs, reward, done, truncate, next_state, final_state



class ParallelEnv:
    def __init__(self, json_file, num_env, truncate=10, reset_repeat=1, step_repeat=1):
        self.num_env = num_env
        self.json_file = json_file
        self.truncate = truncate
        
        with open(json_file, "r") as file:
            data = json.load(file)
            operations_files = data
        # print(len(operations_files))
        self.operations_files = operations_files

        self.env = Env(
            operations_files=self.operations_files,
            truncate=self.truncate,
            reset_repeat=reset_repeat,
            step_repeat=step_repeat
        )

    def reset(self, idx=None):
        states, observations = [], []
        for _ in range(self.num_env):
            state, obs = self.env.reset(idx=idx)
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

