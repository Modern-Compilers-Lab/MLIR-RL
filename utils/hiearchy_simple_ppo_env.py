from utils.observation_utils import build_nested_loops_feature_vector
from fusion_utils.transforms import (
    transform_dialect_tile,
    transform_dialect_TP,
    transform_dialect_interchange,
    transform_dialect_vectorise,
    transform_dialect_vectorise_img2col,
    apply_conv2d_decomposition,
    get_raw_ast_info,
    get_ast,
    transform_dialect_img2col,
    transform_dialect_prints,
    post_process_transform_dialect_prints
)
from utils.transform_utils import evaluate_code_with_timeout

from data_generation import (
    function_wrapper,
    lower_linalg_to_loops,
    get_nested_loops_data,
)

from utils.consts import (
    MAX_NUM_LOOPS,
    INTERCHANGE_ACTIONS,
    NUM_TILE_SIZES,
    NUM_TRANSFORMATIONS
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
import string

def generate_random_string():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))



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
    
    tmp_file = state.tmp_file
    
    code = code.strip()
    code = code.replace("module {\n", "")
    
    if transformation == 'tiling':
        new_code = transform_dialect_tile(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'parallelization':
        new_code = transform_dialect_TP(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'interchange':
        new_code = transform_dialect_interchange(code, state.operation_tag, parameters, tmp_file)
    elif transformation == 'img2col':
        new_code = transform_dialect_img2col(code, state.operation_tag, tmp_file)
    elif transformation == 'vectorization':
        if state.operation_type == 'conv_2d+img2col':
            new_code = transform_dialect_vectorise_img2col(code, state.operation_tag, tmp_file)
        else:
            new_code = transform_dialect_vectorise(code, state.operation_tag, tmp_file)
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





@dataclass
class OperationState:
    operation_tag: str
    raw_operation: str
    operation_type: str
    lowered_operation: str
    loops_data: dict
    transformed_code: str
    actions: np.array
    actions_mask: np.array
    step_count: int
    exec_time: float
    root_exec_time: float
    transformation_history: list
    cummulative_reward: float
    tmp_file: str

def get_obs(state: OperationState):
    
    loops_data = build_nested_loops_feature_vector(state.loops_data)
        
    action_history = state.actions.reshape(-1)
    action_mask = state.actions_mask
    
    if state.operation_type == 'matmul':
        operation_type = 0
    elif 'conv_2d' in state.operation_type:
        operation_type = 1
    
    operation_type = np.array([operation_type])
    
    obs = np.concatenate((operation_type, loops_data, action_history, action_mask))
    obs[:7] = obs[:7] / 100
    return obs

def initialize_action_mask(action_mask, num_loops, operation_type):
    """
    Action mask (5 + L + L + (L-1) + (L-2) + (L-3) ):
        Transformations: end, TP, T, Interchange
        TP: L loops
        T : L loops
        Interchange: 2-consecutive interchanges: L - 1
                   : 3-consecutive interchanges: L - 2
                   : 4-consecutive interchanges: L - 3
        Interchange: 3L - 6
        
    action_mask[:5] = [end, TP, T, I, Img2Col]
    """
    L = MAX_NUM_LOOPS
    
    TP_BEGIN = 5
    T_BEGIN = TP_BEGIN + L
    I_BEGIN_2C = T_BEGIN + L
    I_BEGIN_3C = I_BEGIN_2C + (L-1)
    I_BEGIN_4C = I_BEGIN_3C + (L-2)
    
    if operation_type == 'conv_2d':
        action_mask[:5] = [False, False, False, False, True]
    else:
        action_mask[:5] = [False, True, False, False, False]
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
    actions_mask: (NUM_TRANSFORMATIONS + L + L + (L-1) + (L-2) + (L-3) )
    action_mask[:NUM_TRANSFORMATIONS] = [end, TP, T, I, Img2Col]
    """
    
    L = MAX_NUM_LOOPS
    
    TP_BEGIN = NUM_TRANSFORMATIONS
    T_BEGIN = TP_BEGIN + L
    I_BEGIN_2C = T_BEGIN + L
    I_BEGIN_3C = I_BEGIN_2C + (L-1)
    I_BEGIN_4C = I_BEGIN_3C + (L-2)
    
    actions_mask = state.actions_mask
    
    
    if transformation == 'img2col':actions_mask[:NUM_TRANSFORMATIONS] = [False, True, False, False, False]
    
    if state.operation_type == "pooling" or state.operation_type == "conv_2d":
        if transformation == 'parallelization':actions_mask[:NUM_TRANSFORMATIONS] = [False, False, True, False, False]
        if transformation == 'tiling':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
        
    elif state.operation_type == "conv_2d+img2col":
        if transformation == 'parallelization':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
    
    elif state.operation_type == "matmul":
        if transformation == 'parallelization':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
        if transformation == 'tiling':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, True, True, False]
        if transformation == 'interchange':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, True, False]
        
    else:
        if transformation == 'parallelization':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
        # if transformation == 'interchange':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, True, True, False]
        if transformation == 'tiling':actions_mask[:NUM_TRANSFORMATIONS] = [True, False, True, False, False]
    
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
        # step = len(div) // num_candidates 
        
        step = 1
        if num_candidates <= (len(div) // 2):
            step = 2
        
        res = div[::step][:num_candidates]
    else:
        res = div + div[-1:]*(num_candidates-len(div))
    return res

def last_tiling(history):
    for transformation, parameters in history[::-1]:
        if transformation in ['tiling', 'parallelization']:
            return parameters
    return None
    

def process_action(action_index, state: OperationState):

    loops_data = state.loops_data
    num_loop = len(loops_data['nested_loops'])
    action_name, parameter = action_index

    candidates = [[0]+get_candidates(upper, num_candidates=NUM_TILE_SIZES) for (arg, lower, upper, step) in loops_data['nested_loops']]
    
    if action_name == 'interchange': # interchange
        parameters = INTERCHANGE_ACTIONS[parameter]
        parameters = parameters[:num_loop]
        return ['interchange', list(parameters)]
    
    elif action_name == 'img2col': # interchange
        return ['img2col', [0]]
    
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
        
        operations = [
            'linalg.matmul',
            # 'linalg.conv_2d',
            # 'generic',
            # 'pool',
        ]
        operations_files = { file:details for file, details in operations_files.items() if any([ s in file for s in operations ]) }
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
        
        
        # Generate a random file to apply the transformations and evaluate the code
        random_str = generate_random_string()
        tmp_file = "/scratch/nb3891/Script/MLIR_RL_2/tmp_files/" + random_str + ".txt"
        with open(tmp_file, "w") as file:
            file.write("")
        self.tmp_file = tmp_file


            
    def reset(self, idx=None):
        operations_files = self.operations_files
        
        if idx is not None:
            raw_operation, operation_dict = operations_files[idx]
        else:
            raw_operation, operation_dict = random.choice(operations_files)
            
        num_loops = len(operation_dict["loops_data"]["nested_loops"]) 

        # exec_time = evaluate_code_2(
        #     code=operation_dict["transform_wrapped_operation"],
        # )
        
        exec_time = operation_dict['execution_time']

        # operation_type:
        if 'linalg.matmul' in raw_operation:
            operation_type = 'matmul'
        elif 'linalg.conv' in raw_operation:
            operation_type = 'conv_2d'
       
        # Action mask:
        # Transformations: TP, T, Interchange
        # TP: L loops
        # T : L loops
        # Interchange: 2-consecutive interchanges: L - 1
        #            : 3-consecutive interchanges: L - 2
        #            : 4-consecutive interchanges: L - 3
        # Interchange: 3L - 6
        actions_mask = np.ones((5 + MAX_NUM_LOOPS + MAX_NUM_LOOPS + 3*MAX_NUM_LOOPS - 6), dtype=np.bool_)
        actions_mask = initialize_action_mask(actions_mask, num_loops, operation_type)

        # Action history:
        # 3 because we have 3 transformations: TP, T, I
        actions = np.zeros((MAX_NUM_LOOPS, 3, self.truncate,))


        state = OperationState(
            operation_tag=operation_dict["operation_tag"],
            raw_operation=raw_operation,
            operation_type=operation_type,
            lowered_operation=operation_dict["lowered_operation"],
            loops_data=operation_dict["loops_data"],
            transformed_code=operation_dict["transform_wrapped_operation"],
            # actions=np.zeros((self.truncate, len(PPO_ACTIONS))),
            actions=actions,
            actions_mask=actions_mask,
            step_count=0,
            exec_time=exec_time,
            root_exec_time=exec_time,
            transformation_history=[],
            cummulative_reward=0,
            tmp_file=self.tmp_file
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

        reward = 0
        time_out = False

        
        if transformation != 'no_transformation':
            
            transformed_code = apply_transformation_with_timeout(
                state=state,
                code=state.transformed_code,
                transformation=transformation,
                parameters=parameters,
                timeout=20,
            )
            
            if (transformation == 'tiling') and (state.operation_type == 'conv_2d' or 'pooling' in state.operation_type):
                    
                    if ('conv_2d_nhwc_hwcf' in state.raw_operation):
                        second_interchange_parameters = parameters.copy()
                        second_interchange_parameters[1] = 1
                        second_interchange_parameters[4] = 1
                        
                    elif ('conv_2d_nchw_fchw' in state.raw_operation):
                        second_interchange_parameters = parameters.copy()
                        second_interchange_parameters[2] = 1
                        second_interchange_parameters[5] = 1
                        
                    elif ('pooling' in state.raw_operation):
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
            
            
            if (transformation == 'img2col') and (state.operation_type == 'conv_2d'):
                
                prints = transform_dialect_prints(transformed_code, [state.operation_tag])
                prints = post_process_transform_dialect_prints(prints)
                raw_operation = list(prints.values())[0]
                
                wrapped_operation = function_wrapper(raw_operation)  
                loops = lower_linalg_to_loops(wrapped_operation)            
                loops_data = get_nested_loops_data(loops)
                                                
                state = OperationState(
                    operation_tag=state.operation_tag,
                    raw_operation=raw_operation,
                    operation_type='conv_2d+img2col',
                    lowered_operation=state.lowered_operation,
                    loops_data=loops_data, #
                    transformed_code=state.transformed_code,
                    actions=state.actions,
                    actions_mask=state.actions_mask,
                    step_count=state.step_count + 1,
                    exec_time=state.exec_time,
                    root_exec_time=state.root_exec_time,
                    transformation_history=state.transformation_history + [(transformation, parameters)],
                    cummulative_reward=state.cummulative_reward,
                    tmp_file=self.tmp_file
                )
            
            
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
                    


        next_state = OperationState(
            operation_tag=state.operation_tag,
            raw_operation=state.raw_operation,
            operation_type=state.operation_type,
            lowered_operation=state.lowered_operation,
            loops_data=state.loops_data,
            transformed_code=transformed_code,
            actions=next_state_actions,
            actions_mask=new_actions_mask,
            step_count=state.step_count + 1,
            exec_time=new_exec_time,
            root_exec_time=state.root_exec_time,
            transformation_history=state.transformation_history + [(transformation, parameters)],
            cummulative_reward=state.cummulative_reward,
            tmp_file=self.tmp_file
        )

        next_obs = self.get_obs(next_state)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_obs = torch.unsqueeze(next_obs, 0)

        done = truncate = (next_state.step_count >= self.truncate) or \
            (transformation == 'no_transformation') or \
            (transformation == 'vectorization') or \
            (time_out)


        if done:
            
            if (state.operation_type == 'conv_2d') or ('pooling' in state.operation_type):
                next_state.transformed_code = apply_conv2d_decomposition(next_state.transformed_code, next_state.operation_tag)           

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
                new_exec_time = evaluate_code_with_timeout(code=vect_transformed_code, timeout=120, tmp_file=self.tmp_file)
                # new_exec_time = random.random()
            if new_exec_time is not None:
                r = speedup_reward(new_exec_time, next_state.root_exec_time)
                
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
            next_obs, reward, done, truncate, next_state, final_state = self.env.step(state, action, model)
            batch_next_obs.append( next_obs )
            batch_reward.append( reward )
            batch_done.append( done )
            batch_truncate.append( truncate )
            batch_next_state.append( next_state )
            batch_final_state.append( final_state )

        return batch_next_obs, batch_reward, batch_done, batch_truncate, batch_next_state, batch_final_state

