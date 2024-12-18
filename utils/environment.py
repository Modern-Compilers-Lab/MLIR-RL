from utils.observation_utils import (
    build_nested_loops_feature_vector,
    function_wrapper,
    lower_linalg_to_loops,
)
from utils.transforms import (
    apply_conv2d_decomposition,
    get_raw_ast_info,
    get_ast,
    transform_dialect_prints,
    post_process_transform_dialect_prints,
    evaluate_code_with_timeout,
    apply_transformation_with_timeout,
    extract_loops_data_from_file,
    extract_loops_data_from_code,
)
from utils.lqcd_runner import lower_and_run_code
from typing import Optional

from data_generation_random import get_nested_loops_data

from utils.consts import (
    MAX_NUM_LOOPS,
    NUM_TILE_SIZES,
    NUM_TRANSFORMATIONS
)

import torch

import numpy as np
import random
import json
from dataclasses import dataclass
from tqdm import tqdm
import string
import math
import os

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


@dataclass
class OperationState:
    bench_name: Optional[str]
    operation_tag: str
    raw_operation: str
    operation_type: str
    # lowered_operation: str
    loops_data: dict
    transformed_code: str
    actions: np.array
    actions_mask: np.array
    step_count: int
    exec_time: float
    root_exec_time: float
    transformation_history: list
    bench_transformation_history: list
    cummulative_reward: float
    tmp_file: str

def get_obs(state: OperationState):
    """
    Build the obervation vector for the input state
    """

    loops_data = build_nested_loops_feature_vector(state.loops_data)

    action_history = state.actions.reshape(-1)
    action_mask = state.actions_mask

    if state.operation_type == 'matmul':
        operation_type = 0
    elif 'conv_2d' in state.operation_type:
        operation_type = 1
    elif state.operation_type == 'pooling':
        operation_type = 2
    elif state.operation_type == 'add':
        operation_type = 3
    elif state.operation_type == 'generic':
        operation_type = 4

    operation_type = np.array([operation_type])

    obs = np.concatenate((
        # The input of the policy network:
        operation_type,  # 1
        loops_data,      # MAX_NUM_LOOPS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM*MAX_NUM_STORES_LOADS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM + 5
        action_history,  # MAX_NUM_LOOPS*3*CONFIG["truncate"]

        # The action mask:
        action_mask     # 5 + MAX_NUM_LOOPS + MAX_NUM_LOOPS + (MAX_NUM_LOOPS-1) + (MAX_NUM_LOOPS-2) + (MAX_NUM_LOOPS-3)
    ))

    # Normalize the upper bounds of the loops
    obs[1:MAX_NUM_LOOPS + 1] = obs[1:MAX_NUM_LOOPS + 1] / 100

    return obs

def initialize_action_mask(num_loops, operation_type):
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
    I_BEGIN_3C = I_BEGIN_2C + (L - 1)
    I_BEGIN_4C = I_BEGIN_3C + (L - 2)

    action_mask = np.ones((5 + MAX_NUM_LOOPS + MAX_NUM_LOOPS + 3 * MAX_NUM_LOOPS - 6), dtype=np.bool_)
    if operation_type == 'conv_2d':
        action_mask[:5] = [False, False, False, False, True]
    else:
        action_mask[:5] = [False, True, False, False, False]
        # action_mask[:5] = [False, True, True, True, False]
    action_mask[TP_BEGIN + num_loops:T_BEGIN] = False
    action_mask[T_BEGIN + num_loops:I_BEGIN_2C] = False
    action_mask[I_BEGIN_2C + num_loops - 1:I_BEGIN_3C] = False
    action_mask[I_BEGIN_3C + num_loops - 2:I_BEGIN_4C] = False
    action_mask[I_BEGIN_4C + num_loops - 3:] = False

    if num_loops == 1:
        action_mask[3] = False
        action_mask[I_BEGIN_2C] = True

    return action_mask

def update_action_mask(state, transformation, num_loops):
    """
    actions_mask: (NUM_TRANSFORMATIONS + L + L + (L-1) + (L-2) + (L-3) )
    action_mask[:NUM_TRANSFORMATIONS] = [end, TP, T, I, Img2Col]
    """

    L = MAX_NUM_LOOPS

    TP_BEGIN = NUM_TRANSFORMATIONS
    T_BEGIN = TP_BEGIN + L
    I_BEGIN_2C = T_BEGIN + L
    # I_BEGIN_3C = I_BEGIN_2C + (L-1)
    # I_BEGIN_4C = I_BEGIN_3C + (L-2)

    actions_mask = state.actions_mask

    if transformation == 'img2col':
        actions_mask[:NUM_TRANSFORMATIONS] = [False, True, False, False, False]

    if state.operation_type == "pooling" or state.operation_type == "conv_2d":
        if transformation == 'parallelization':
            actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
        if transformation == 'tiling':
            actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]

    elif state.operation_type == "conv_2d+img2col":
        if transformation == 'parallelization':
            actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]

    elif state.operation_type == "matmul" or state.operation_type == "add":
        if transformation == 'parallelization':
            actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
        if transformation == 'tiling':
            actions_mask[:NUM_TRANSFORMATIONS] = [True, False, True, True, False]
        if transformation == 'interchange':
            actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, True, False]

    elif state.operation_type == "generic":
        if transformation == 'parallelization':
            actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
        if transformation == 'interchange':
            # NOTE: actions_mask[:NUM_TRANSFORMATIONS] = [True, False, True, True, False]
            actions_mask[:NUM_TRANSFORMATIONS] = [True, True, True, True, False]
        if transformation == 'tiling':
            # NOTE: actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
            actions_mask[:NUM_TRANSFORMATIONS] = [True, True, True, True, False]

    else:
        raise ValueError("operation_type must be in [pooling, conv_2d, conv_2d+img2col, matmul, add, generic]")

    if num_loops == 1:
        actions_mask[3] = False
        actions_mask[I_BEGIN_2C] = True

    return actions_mask

def update_action_history(state: OperationState, transformation, parameters):
    # actions.shape: (L, 3, truncate)
    # parallelization, tiling, interchange

    num_loops = len(state.loops_data["nested_loops"])
    actions = state.actions
    assert state.step_count < state.actions.shape[2]

    # actions[l, t, s] = the parameters of transformation `t` for loop `l` at step `s`
    for loop_index in range(num_loops):
        if transformation == 'parallelization':
            actions[loop_index, 0, state.step_count] = parameters[loop_index]
        elif transformation == 'tiling':
            actions[loop_index, 1, state.step_count] = parameters[loop_index]
        elif transformation == 'interchange':
            actions[loop_index, 2, state.step_count] = parameters[loop_index]

    return actions

def get_interchange_actions(num_loops: int):
    """
    Get all the possible interchanges for `num_loops`
    """

    interchanges = []
    for c in [1, 2, 3]:
        level_interchanges = []
        for _ in range(MAX_NUM_LOOPS - c):
            level_interchanges.append(tuple(range(num_loops)))
        for i in range(num_loops - c):
            params = list(range(num_loops))
            params[i], params[i + c] = params[i + c], params[i]
            level_interchanges[i] = tuple(params)
        interchanges += level_interchanges
    return interchanges

def sorted_divisors(n, num_candidates):
    """
    Get the divisors of `n` that are supperior or equal to 2
    """

    divisors = []
    i = 1
    while i <= n and len(divisors) < num_candidates:
        if n % i == 0:
            divisors.append(i)
        i *= 2
    return sorted(divisors)

def get_tiling_candidates(n, num_candidates):
    """
    Get `num_candidates` candidate tiling size for upper bound `n`
    """

    # If upperbound equal 1, we only have candidates of 1
    if n == 1:
        return [1] * num_candidates

    # We take the divisors of the upperbound `n`
    div = sorted_divisors(n, num_candidates)

    if len(div) < num_candidates:  # If we don't have enough unique divisors, we fill the rest of the candidates with the last dividor
        res = div + div[-1:] * (num_candidates - len(div))
    else:
        res = div[:num_candidates]
    return res

def last_tiling(history):
    """
    Get the last tiling from the action history
    """
    for transformation, parameters in history[::-1]:
        if transformation in ['tiling', 'parallelization']:
            return parameters
    return None


def process_action(raw_action, state: OperationState):
    """
    Get the (transformation, parameters) from the `raw_action`.
    """

    loops_data = state.loops_data
    num_loops = len(loops_data['nested_loops'])
    action_name, parameter = raw_action

    # Sellect the tiling candidates for each loop
    if action_name in ['tiling', 'parallelization']:
        candidates = [
            [0] + get_tiling_candidates(upper, num_candidates=NUM_TILE_SIZES)
            for (_, _, upper, _, _) in loops_data['nested_loops']
        ]

    if action_name == 'interchange':
        candidates = get_interchange_actions(num_loops)
        parameters = candidates[parameter]
        assert len(parameters) == num_loops
        return ['interchange', list(parameters)]

    elif action_name == 'img2col':
        return ['img2col', [0]]

    elif action_name == 'tiling':
        tiling_parameters = []
        for i in range(num_loops):
            if i < len(parameter):
                if parameter[i] != -1:
                    tiling_parameters.append(candidates[i][parameter[i]])
                else:  # parameter[i] == -1:
                    tiling_parameters.append(0)
            else:  # i >= len(parameter)
                tiling_parameters.append(0)

        last_tiling_parameters = last_tiling(state.transformation_history)
        if last_tiling_parameters is not None:
            tiling_parameters = [a if (a == 0) or ((a != 0) and (b % a == 0)) else b for a, b in zip(tiling_parameters, last_tiling_parameters)]

        return ['tiling', tiling_parameters]

    elif action_name == 'parallelization':
        parall_parameters = []
        for i in range(num_loops):
            if i < len(parameter):
                if parameter[i] != -1:
                    parall_parameters.append(candidates[i][parameter[i]])
                else:  # parameter[i] == -1:
                    parall_parameters.append(0)
            else:  # i >= len(parameter)
                parall_parameters.append(0)

        return ['parallelization', parall_parameters]

    return ['no_transformation', [0]]


def speedup_reward(new, old, a=10):

    # if old >= new:
    #     reward = old/new - 1
    # else: # old < new
    #     reward = - new/old + 1

    # reward = math.log(old / new) / math.log(a)
    reward = math.log(old / new, a)

    return reward


class Env:
    def __init__(self, json_file, truncate=10, reset_repeat=1, step_repeat=1, tmp_file: str = None, from_lqcd=True):
        # Generate a random file to be used in order to apply the transformations and evaluate the code
        # This is done in order to enable having multiple experiments at the same time, by letting each
        # experiment use a separate unique file to read and write intermidiate representations
        random_str = generate_random_string()
        if tmp_file is None:
            tmp_file = f"tmp_files/{random_str}.mlir"
        with open(tmp_file, "w") as file:
            file.write("")
        self.tmp_file = tmp_file

        self.from_lqcd = from_lqcd
        if from_lqcd:
            with open(json_file, "r") as file:
                benchmarks_json: dict[str, float] = json.load(file)
            benchmarks_data: list[tuple[str, dict]] = []
            for bench_name, exec_time in benchmarks_json.items():
                bench_file = os.path.join("lqcd-benchmarks", bench_name + ".mlir")
                benchmark_data = extract_loops_data_from_file(bench_file, exec_time)
                benchmarks_data.append((bench_name, benchmark_data))
            self.operations_files = benchmarks_data
        else:
            with open(json_file, "r") as file:
                operations_files = json.load(file)
            operations = [
                # 'linalg.matmul',
                # 'linalg.conv_2d',
                # 'pooling',
                'generic',
                # 'linalg.add',
            ]
            operations_files = {file: details for file, details in operations_files.items() if any([s in file for s in operations])}
            operations_files = [[details['operation'], details] for file, details in operations_files.items()]

            # Get the AST of the MLIR code and give a tag to each linalg operation
            # The last operation represents the operations that we want to optimize (the first operations are just linalg.fills)
            for i in tqdm(range(len(operations_files))):
                code = operations_files[i][1]["transform_wrapped_operation"]
                raw_ast_info = get_raw_ast_info(code, self.tmp_file)
                code_ast, code_with_tags = get_ast(raw_ast_info)
                operations_files[i][1]["transform_wrapped_operation"] = code_with_tags

                # We take  the operation with the last tag
                operation_tag = list(code_ast.keys())[-1]
                # TODO: Needs further verification
                operations_files[i][1]["operation_tag"] = operation_tag

            self.operations_files = operations_files

        self.truncate = truncate
        self.get_obs = get_obs
        self.reset_repeat = reset_repeat
        self.step_repeat = step_repeat

    def reset(self, idx=None):
        if idx is not None:
            # We get the operation with the right index
            self.bench_index = idx
        else:
            # Get a random operation
            self.bench_index = random.randint(0, len(self.operations_files) - 1)
        raw_operation, operation_dict = self.operations_files[self.bench_index]

        # The baseline execution time of the Linalg operation
        exec_time = operation_dict['execution_time']

        # The number of loops in the Linalg operations
        if self.from_lqcd:
            bench_name = raw_operation
            bench_file = os.path.join("lqcd-benchmarks", bench_name + ".mlir")
            with open(bench_file, "r") as file:
                code = file.read()
            real_exec_time, _ = lower_and_run_code(code, bench_name)
            print_info(f"Real exec time: {real_exec_time}, expected exec time: {exec_time}")
            lqcd_operation_tag = operation_dict["ops_tags"][0]
            lqcd_operation_dict = operation_dict[lqcd_operation_tag]
            num_loops = len(lqcd_operation_dict["nested_loops"])
        else:
            num_loops = len(operation_dict["loops_data"]["nested_loops"])

        # operation_type:
        if self.from_lqcd:
            operation_type = 'generic'
        else:
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

        # Action mask:
        # Transformations: 5 = TP, T, Interchange, Im2col, Vectorization
        # TP: L loops
        # T : L loops
        # Interchange: 3L - 6 (total)
        #            : L - 1 for 2-consecutive interchanges
        #            : L - 2 for 3-consecutive interchanges
        #            : L - 3 for 4-consecutive interchanges
        actions_mask = initialize_action_mask(num_loops, operation_type)

        # Action history:
        # 3 because we have 3 transformations that require parameters: TP, T, I
        actions = np.zeros((MAX_NUM_LOOPS, 3, self.truncate,))

        state = OperationState(
            bench_name=bench_name if self.from_lqcd else None,
            operation_tag=lqcd_operation_tag if self.from_lqcd else operation_dict["operation_tag"],
            raw_operation=lqcd_operation_dict["operation"] if self.from_lqcd else raw_operation,
            operation_type=operation_type,
            # lowered_operation=operation_dict["lowered_operation"],
            loops_data=lqcd_operation_dict if self.from_lqcd else operation_dict["loops_data"],
            transformed_code=operation_dict["transform_wrapped_operation"],
            actions=actions,
            actions_mask=actions_mask,
            step_count=0,
            exec_time=exec_time,
            root_exec_time=exec_time,
            transformation_history=[],
            bench_transformation_history=[],
            cummulative_reward=0,
            tmp_file=self.tmp_file
        )

        obs = self.get_obs(state)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = torch.unsqueeze(obs, 0)

        return state, obs

    def step(self, state, raw_action) -> tuple[np.ndarray, float, bool, OperationState, Optional[OperationState]]:
        """
        raw_action: (action:str, parameters:list[int])
        """

        # The number of loops in the Linalg operations
        num_loops = len(state.loops_data["nested_loops"])

        # preprocess the action coming from the policy network and make it more explicit
        # aka get the transformation and its parameters (equal to None if no parameters are needded)
        transformation, parameters = process_action(
            raw_action=raw_action,
            state=state
        )

        print_info("RAW:", raw_action)
        print_success("PROCESSED:", transformation, parameters)

        reward = 0

        if transformation not in ['no_transformation', 'vectorization']:
            # Apply the transformation and get the new code
            transformed_code = apply_transformation_with_timeout(
                state=state,
                code=state.transformed_code,
                transformation=transformation,
                parameters=parameters,
                timeout=20,
                from_lqcd=self.from_lqcd
            )

            # SPECIAL CASE:
            # If we are optimizing a convlution operation, then we can apply the Im2col transformation to turn the convolution
            # into a matrix multiplicatoin.
            if transformed_code and (transformation == 'img2col') and (state.operation_type == 'conv_2d'):

                # Get the matmul operation that now represents the convlution and wrap it in a funciton wrapper
                # to prepare it for the optimization in the next iterations

                prints = transform_dialect_prints(transformed_code, [state.operation_tag], self.tmp_file)
                prints = post_process_transform_dialect_prints(prints)
                raw_operation = list(prints.values())[0]

                wrapped_operation = function_wrapper(raw_operation)
                loops = lower_linalg_to_loops(wrapped_operation, self.tmp_file)
                loops_data = get_nested_loops_data(loops)

                state = OperationState(
                    bench_name=state.bench_name,
                    operation_tag=state.operation_tag,
                    raw_operation=raw_operation,
                    operation_type='conv_2d+img2col',  # The operation type changes
                    # lowered_operation=state.lowered_operation,
                    loops_data=loops_data,  # The loops changed because now we are optimization a mamtul instead of a convolution
                    transformed_code=state.transformed_code,
                    actions=state.actions,
                    actions_mask=state.actions_mask,
                    step_count=state.step_count + 1,
                    exec_time=state.exec_time,
                    root_exec_time=state.root_exec_time,
                    transformation_history=state.transformation_history + [(transformation, parameters)],
                    bench_transformation_history=state.bench_transformation_history + [(transformation, parameters)],
                    cummulative_reward=state.cummulative_reward,
                    tmp_file=self.tmp_file
                )

        else:  # transformation == 'no_transformation' or 'vectorization'
            # For convolution, before vectorization, we need to first apply another tiling in order to decompose it to 1d convolution
            if (state.operation_type == 'conv_2d'):
                if ('conv_2d_nhwc_hwcf' in state.raw_operation):
                    second_interchange_parameters = parameters.copy()
                    second_interchange_parameters[1] = 1
                    second_interchange_parameters[4] = 1
                elif ('conv_2d_nchw_fchw' in state.raw_operation):
                    second_interchange_parameters = parameters.copy()
                    second_interchange_parameters[2] = 1
                    second_interchange_parameters[5] = 1
                elif ('pooling' in state.raw_operation):
                    second_interchange_parameters = [0] * 6
                    second_interchange_parameters[2] = 1
                    second_interchange_parameters[4] = 1
                state.transformed_code = apply_transformation_with_timeout(
                    state=state,
                    code=state.transformed_code,
                    transformation='tiling',
                    parameters=second_interchange_parameters,
                    timeout=20,
                    from_lqcd=self.from_lqcd
                )

                state.transformed_code = apply_conv2d_decomposition(state.transformed_code, state.operation_tag, self.tmp_file)

            # Generic and pooling operations are better without vectorization
            if state.operation_type != 'pooling':
                # Apply the vectorization and get the new code
                transformation = 'vectorization'
                transformed_code = apply_transformation_with_timeout(
                    state=state,
                    code=state.transformed_code,
                    transformation=transformation,
                    parameters=parameters,
                    timeout=20,
                    from_lqcd=self.from_lqcd
                )
            else:
                transformation = 'no_transformation'
                transformed_code = state.transformed_code

        trans_failed = not transformed_code  # This indicatesthat that the transformation failed or timed out
        if trans_failed:
            # We keep the same code as previously
            # We get a penalty of -5
            print_error(f'FAILED TRANSFORM: {transformation} {parameters} {state.transformation_history}')
            transformed_code = state.transformed_code
            reward -= 5

        # Update state actions:
        next_state_actions = update_action_history(state, transformation, parameters)

        # Update action mask:
        new_actions_mask = update_action_mask(state, transformation, num_loops)

        next_state = OperationState(
            bench_name=state.bench_name,
            operation_tag=state.operation_tag,
            raw_operation=state.raw_operation,
            operation_type=state.operation_type,
            # lowered_operation=state.lowered_operation,
            loops_data=state.loops_data,
            transformed_code=transformed_code,  # New transformed code
            actions=next_state_actions,  # New actions
            actions_mask=new_actions_mask,  # New action mask
            step_count=state.step_count + 1,
            exec_time=state.exec_time,  # New execution time
            root_exec_time=state.root_exec_time,
            transformation_history=state.transformation_history + [(transformation, parameters)],
            bench_transformation_history=state.bench_transformation_history + [(transformation, parameters)],
            cummulative_reward=state.cummulative_reward,
            tmp_file=self.tmp_file
        )

        # Done == True if:
        #   We surpass the maximum number of steps (size of the schedule)
        #   Vectorization indicating the end of the schedule
        #   Error occured in the transformation
        done = (next_state.step_count >= self.truncate) or \
            (transformation in ['vectorization', 'no_transformation']) or \
            (trans_failed)
        should_reset_if_done = True

        if done:
            if self.from_lqcd:
                assert next_state.bench_name is not None
                new_exec_time, bench_passed_or_exception = lower_and_run_code(transformed_code, next_state.bench_name)
                if new_exec_time is None:
                    print_error(f"EXECUTION ERROR: {bench_passed_or_exception}")
                elif not bench_passed_or_exception:
                    reward -= 20
                    print_error("ASSERTION FAILED")
            else:
                new_exec_time = evaluate_code_with_timeout(transformed_code, 120, self.tmp_file)

            if new_exec_time is not None:  # If the code has been executed successfuly and we have an execution time
                # We calculate the speedup
                reward += speedup_reward(new_exec_time, next_state.root_exec_time)
                next_state.exec_time = new_exec_time
            else:
                reward -= 20
                print_error(f'EVAL ERROR: {transformation} {parameters} {next_state.transformation_history}')
                new_exec_time = next_state.exec_time

            if self.from_lqcd:
                lqcd_bench_name, lqcd_bench_dict = self.operations_files[self.bench_index]
                lqcd_op_index = lqcd_bench_dict["ops_tags"].index(next_state.operation_tag)
                if lqcd_op_index < len(lqcd_bench_dict["ops_tags"]) - 1:
                    # Indicates that the trajectory isn't over yet, so don't reset
                    should_reset_if_done = False

                    speedup_metric = next_state.root_exec_time / next_state.exec_time
                    print('-' * 30)
                    print(f"Operation: {next_state.bench_name} - {next_state.operation_tag}")
                    print(next_state.transformation_history)
                    print('Speedup:', speedup_metric)
                    print('Old Exec time:', next_state.root_exec_time, 's')
                    print('New Exec time:', next_state.exec_time, 's')
                    print('-' * 30)

                    # Re-extract operations data from the new code
                    new_bench_dict = extract_loops_data_from_code(next_state.transformed_code, next_state.exec_time)
                    self.operations_files[self.bench_index] = (lqcd_bench_name, new_bench_dict)

                    # Build a new state that points to the next operation
                    new_op_tag = lqcd_bench_dict["ops_tags"][lqcd_op_index + 1]
                    new_op_dict = new_bench_dict[new_op_tag]
                    actions_mask = initialize_action_mask(len(new_op_dict['nested_loops']), next_state.operation_type)
                    next_state = OperationState(
                        bench_name=lqcd_bench_name,
                        operation_tag=new_op_tag,
                        raw_operation=new_op_dict["operation"],
                        operation_type=next_state.operation_type,
                        loops_data=new_op_dict,
                        transformed_code=new_bench_dict["transform_wrapped_operation"],
                        actions=np.zeros((MAX_NUM_LOOPS, 3, self.truncate,)),
                        actions_mask=actions_mask,
                        step_count=0,
                        exec_time=next_state.exec_time,
                        root_exec_time=next_state.root_exec_time,
                        transformation_history=[],
                        bench_transformation_history=next_state.bench_transformation_history.copy(),
                        cummulative_reward=next_state.cummulative_reward,
                        tmp_file=self.tmp_file
                    )

        next_state.cummulative_reward += reward

        next_obs = self.get_obs(next_state)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        next_obs = torch.unsqueeze(next_obs, 0)

        final_state = None
        if done and should_reset_if_done:
            final_state = next_state
            next_state, next_obs = self.reset()

        return next_obs, reward, done, next_state, final_state


class ParallelEnv:
    def __init__(self, json_file, num_env=1, truncate=10, reset_repeat=1, step_repeat=1, from_lqcd=True):
        self.num_env = num_env
        tmp_files = [f"tmp_files/tmp_{i}.mlir" for i in range(num_env)]

        self.env = Env(
            json_file=json_file,
            truncate=truncate,
            reset_repeat=reset_repeat,
            step_repeat=step_repeat,
            tmp_file=tmp_files[0],
            from_lqcd=from_lqcd
        )

    def reset(self, idx=None):
        states, observations = [], []
        for _ in range(self.num_env):
            state, obs = self.env.reset(idx=idx)
            states.append(state)
            observations.append(obs)
        return states, observations

    def step(self, states, actions) -> tuple[list[np.ndarray], list[float], list[bool], list[OperationState], list[Optional[OperationState]]]:
        batch_next_obs, batch_reward, batch_done, batch_next_state, batch_final_state = [], [], [], [], []
        for state, action in zip(states, actions):
            next_obs, reward, done, next_state, final_state = self.env.step(state, action)

            batch_next_obs.append(next_obs)
            batch_reward.append(reward)
            batch_done.append(done)
            batch_next_state.append(next_state)
            batch_final_state.append(final_state)

        return batch_next_obs, batch_reward, batch_done, batch_next_state, batch_final_state
