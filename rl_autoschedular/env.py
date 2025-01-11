import torch
import numpy as np
import random
import json
from tqdm import tqdm
import math
import os
import string
from typing import Optional, Literal
from rl_autoschedular import config as cfg
from rl_autoschedular.state import OperationState, BenchmarkFeatures
from rl_autoschedular.observation import (
    extract_bench_features_from_file,
    extract_bench_features_from_code,
    extract_op_features_from_affine_code,
    build_op_features_vector
)
from rl_autoschedular.transforms import (
    apply_transformation_with_timeout,
    get_ops_by_tags,
    apply_conv2d_decomposition
)
from rl_autoschedular.evaluation import (
    evaluate_code_with_bindings_and_timeout,
    evaluate_code_with_cmd_and_timeout
)
from utils.log import print_info, print_success, print_error


class Env:
    """Environment for training the reinforcement learning agent."""

    benchmarks_data: list[tuple[str, BenchmarkFeatures]]
    """Lists for each benchmark the benchmark's name and its features."""
    truncate: int
    """The maximum number of steps in the schedule."""
    reset_repeat: int
    """The number of times to repeat the reset function."""
    step_repeat: int
    """The number of times to repeat the step function."""
    tmp_file: str
    """The temporary file to store the intermediate representations."""

    def __init__(self, reset_repeat: int = 1, step_repeat: int = 1, tmp_file: Optional[str] = None):
        """Initialize the environment.

        Args:
            reset_repeat (int): The number of times to repeat the reset function. Defaults to 1.
            step_repeat (int): The number of times to repeat the step function. Defaults to 1.
            tmp_file (Optional[str]): The temporary file to store the intermediate representations. Defaults to None.
        """
        # Generate a random file to be used in order to apply the transformations and evaluate the code
        # This is done in order to enable having multiple experiments at the same time, by letting each
        # experiment use a separate unique file to read and write intermediate representations
        random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
        if tmp_file is None:
            tmp_file = f"tmp/{random_str}.mlir"
        with open(tmp_file, "w") as file:
            file.write("")
        self.tmp_file = tmp_file

        # Get benchmarks data
        self.benchmarks_data = []
        if cfg.data_format == "mlir":
            # Load execution times from json file
            with open(cfg.json_file, "r") as file:
                benchmarks_json: dict[str, float] = json.load(file)
            # Build benchmark features
            for bench_name, exec_time in benchmarks_json.items():
                bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, exec_time * 10**9, exec_time * 10**9)
                self.benchmarks_data.append((bench_name, benchmark_data))
        else:
            # Load operations data from json file
            with open(cfg.json_file, "r") as file:
                json_data = json.load(file)
            operation_filter = [
                'linalg.matmul',
                'linalg.conv_2d',
                # 'pooling',
                # 'generic',
                'linalg.add',
            ]
            json_data = {op: details for op, details in json_data.items() if any([s in op for s in operation_filter])}
            json_data = [(details['operation'], details) for _, details in json_data.items()]

            # Get the AST of the MLIR code and give a tag to each linalg operation
            # The last operation represents the operations that we want to optimize (the first operations are just linalg.fills)
            for i in tqdm(range(len(json_data))):
                # Get full MLIR code and execution time
                code = json_data[i][1]["transform_wrapped_operation"]
                exec_time = json_data[i][1]["execution_time"]
                # Build benchmark features
                bench_name = f"bench_{i}"
                benchmark_data = extract_bench_features_from_code(bench_name, code, exec_time, exec_time)
                self.benchmarks_data.append((bench_name, benchmark_data))

        self.reset_repeat = reset_repeat
        self.step_repeat = step_repeat

    def reset(self, idx: Optional[int] = None):
        """Reset the environment.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            OperationState: The initial state of the environment.
            torch.Tensor: The observation vector of the initial state.
        """
        if idx is not None:
            # We get the operation with the right index
            self.bench_index = idx
        else:
            # Get a random operation
            self.bench_index = random.randint(0, len(self.benchmarks_data) - 1)
        bench_name, benchmark_data = self.benchmarks_data[self.bench_index]

        # The number of loops in the Linalg operations
        if cfg.data_format == "mlir":
            # Get benchmark file
            bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
            # Reload original code and features
            benchmark_data = extract_bench_features_from_file(bench_name, bench_file, benchmark_data.root_exec_time, benchmark_data.root_exec_time)
            self.benchmarks_data[self.bench_index] = (bench_name, benchmark_data)
        # TODO: Add case where data_format is "json" and reload data from json file if needed (if optimization mode is "all")

        # Get the last operation
        operation_tag = benchmark_data.operation_tags[-1]
        operation_features = benchmark_data.operations[operation_tag]
        num_loops = len(operation_features.nested_loops)

        # Get operation type
        raw_operation = operation_features.raw_operation
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
        actions_mask = self.initialize_action_mask(num_loops, operation_type)

        # Action history:
        # 3 because we have 3 transformations that require parameters: TP, T, I
        actions = np.zeros((cfg.max_num_loops, 3, cfg.truncate,))

        state = OperationState(
            bench_name=bench_name,
            operation_tag=operation_tag,
            operation_type=operation_type,
            operation_features=operation_features,
            transformed_code=benchmark_data.code,
            actions=actions,
            actions_mask=actions_mask,
            step_count=0,
            exec_time=benchmark_data.exec_time,
            root_exec_time=benchmark_data.exec_time,
            transformation_history=[],
            cummulative_reward=0,
            tmp_file=self.tmp_file
        )

        obs = self.get_obs(state)
        obs = torch.tensor(obs, dtype=torch.float32)
        obs = torch.unsqueeze(obs, 0)

        return state, obs

    def step(self, state: OperationState, raw_action: tuple[str, list[int]]) -> tuple[np.ndarray, float, bool, OperationState, Optional[OperationState]]:
        """Take a step in the environment.

        Args:
            state (OperationState): The current state of the environment.
            raw_action (tuple[str, list[int]]): The raw action taken by the agent. The first element is the transformation name and the second element is the parameters.

        Returns:
            np.ndarray: The observation vector of the next state.
            float: The reward of the action.
            bool: Whether the episode is done.
            OperationState: The next state of the environment.
            Optional[OperationState]: The final state of the environment if the episode is done.
        """

        # The number of loops in the Linalg operations
        num_loops = len(state.operation_features.nested_loops)

        # preprocess the action coming from the policy network and make it more explicit
        # aka get the transformation and its parameters (equal to None if no parameters are needded)
        transformation, parameters = self.process_action(
            raw_action=raw_action,
            state=state
        )

        # Get benchmark data
        bench_name, bench_data = self.benchmarks_data[self.bench_index]

        print_info("RAW:", raw_action)
        print_success("PROCESSED:", transformation, parameters)

        reward = 0
        if transformation not in ['no_transformation', 'vectorization']:
            # Apply the transformation and get the new code
            transformed_code = apply_transformation_with_timeout(
                state=state,
                bench_features=bench_data,
                code=state.transformed_code,
                transformation=transformation,
                parameters=parameters,
                timeout=20,
                use_vectorizer=cfg.use_vectorizer
            )

            # SPECIAL CASE:
            # If we are optimizing a convlution operation, then we can apply the Im2col transformation to turn the convolution
            # into a matrix multiplicatoin.
            if transformed_code and (transformation == 'img2col') and (state.operation_type == 'conv_2d'):

                # Get the matmul operation that now represents the convlution and wrap it in a funciton wrapper
                # to prepare it for the optimization in the next iterations

                prints = get_ops_by_tags(transformed_code, [state.operation_tag], self.tmp_file)
                raw_operation = list(prints.values())[0]

                operation_features = extract_op_features_from_affine_code(raw_operation, self.tmp_file)

                state = OperationState(
                    bench_name=state.bench_name,
                    operation_tag=state.operation_tag,
                    operation_type='conv_2d+img2col',  # The operation type changes
                    operation_features=operation_features,  # The loops changed because now we are optimization a mamtul instead of a convolution
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

        else:  # transformation == 'no_transformation' or 'vectorization'
            # For convolution, before vectorization, we need to first apply another tiling in order to decompose it to 1d convolution
            if (state.operation_type == 'conv_2d'):
                if ('conv_2d_nhwc_hwcf' in state.operation_features.raw_operation):
                    second_interchange_parameters = parameters.copy()
                    second_interchange_parameters[1] = 1
                    second_interchange_parameters[4] = 1
                elif ('conv_2d_nchw_fchw' in state.operation_features.raw_operation):
                    second_interchange_parameters = parameters.copy()
                    second_interchange_parameters[2] = 1
                    second_interchange_parameters[5] = 1
                elif ('pooling' in state.operation_features.raw_operation):
                    second_interchange_parameters = [0] * 6
                    second_interchange_parameters[2] = 1
                    second_interchange_parameters[4] = 1
                state.transformed_code = apply_transformation_with_timeout(
                    state=state,
                    bench_features=bench_data,
                    code=state.transformed_code,
                    transformation='tiling',
                    parameters=second_interchange_parameters,
                    timeout=20,
                    use_vectorizer=cfg.use_vectorizer
                )

                state.transformed_code = apply_conv2d_decomposition(state.transformed_code, state.operation_tag, self.tmp_file)

            # Generic and pooling operations are better without vectorization
            if state.operation_type != 'pooling':
                # Apply the vectorization and get the new code
                transformation = 'vectorization'
                transformed_code = apply_transformation_with_timeout(
                    state=state,
                    bench_features=bench_data,
                    code=state.transformed_code,
                    transformation=transformation,
                    parameters=parameters,
                    timeout=20,
                    use_vectorizer=cfg.use_vectorizer
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
        next_state_actions = self.update_action_history(state, transformation, parameters)

        # Update action mask:
        new_actions_mask = self.update_action_mask(state, transformation, num_loops)

        next_state = OperationState(
            bench_name=state.bench_name,
            operation_tag=state.operation_tag,
            operation_type=state.operation_type,
            operation_features=state.operation_features,
            transformed_code=transformed_code,  # New transformed code
            actions=next_state_actions,  # New actions
            actions_mask=new_actions_mask,  # New action mask
            step_count=state.step_count + 1,
            exec_time=state.exec_time,  # New execution time
            root_exec_time=state.root_exec_time,
            transformation_history=state.transformation_history + [(transformation, parameters)],
            cummulative_reward=state.cummulative_reward,
            tmp_file=self.tmp_file
        )

        # Done == True if:
        #   We surpass the maximum number of steps (size of the schedule)
        #   Vectorization indicating the end of the schedule
        #   Error occured in the transformation
        done = (next_state.step_count >= cfg.truncate) or \
            (transformation in ['vectorization', 'no_transformation']) or \
            (trans_failed)
        should_reset_if_done = True

        if done:
            # Execute and evaluate the code
            if cfg.use_bindings:
                new_exec_time, bench_passed = evaluate_code_with_bindings_and_timeout(transformed_code, next_state.bench_name)
            else:
                new_exec_time, bench_passed = evaluate_code_with_cmd_and_timeout(transformed_code, self.tmp_file, timeout=120)
            # Print infos and update reward
            if new_exec_time is None:
                reward -= 20
                print_error(f"EXECUTION ERROR: {transformation} {parameters} {next_state.transformation_history}")
                new_exec_time = next_state.exec_time
            else:
                if bench_passed:
                    # We calculate the speedup
                    reward += self.speedup_reward(new_exec_time, next_state.root_exec_time)
                    next_state.exec_time = new_exec_time
                else:
                    reward -= 20
                    print_error("ASSERTION FAILED")
                    new_exec_time = next_state.exec_time

            if cfg.optimization_mode == "all":
                op_index = bench_data.operation_tags.index(next_state.operation_tag)
                if op_index > 0:
                    # Indicates that the trajectory isn't over yet, so don't reset
                    should_reset_if_done = False

                    speedup_metric = next_state.root_exec_time / next_state.exec_time
                    print('-' * 30)
                    print(f"Operation: {next_state.bench_name} - {next_state.operation_tag}")
                    print(next_state.transformation_history)
                    print('Relative speedup:', speedup_metric)
                    print('Old Exec time:', next_state.root_exec_time * 10**-9, 's')
                    print('New Exec time:', next_state.exec_time * 10**-9, 's')
                    print(f"reward: {reward}")
                    print(f"cummulative reward: {next_state.cummulative_reward + reward}")

                    # Re-extract operations data from the new code
                    new_bench_data = extract_bench_features_from_code(bench_name, next_state.transformed_code, bench_data.root_exec_time, next_state.exec_time)
                    self.benchmarks_data[self.bench_index] = (bench_name, new_bench_data)

                    # Build a new state that points to the next operation
                    new_op_tag = new_bench_data.operation_tags[op_index - 1]
                    new_op_features = new_bench_data.operations[new_op_tag]
                    actions_mask = self.initialize_action_mask(len(new_op_features.nested_loops), next_state.operation_type)
                    next_state = OperationState(
                        bench_name=bench_name,
                        operation_tag=new_op_tag,
                        operation_type=next_state.operation_type,
                        operation_features=new_op_features,
                        transformed_code=new_bench_data.code,
                        actions=np.zeros((cfg.max_num_loops, 3, cfg.truncate)),
                        actions_mask=actions_mask,
                        step_count=0,
                        exec_time=next_state.exec_time,
                        root_exec_time=next_state.exec_time,
                        transformation_history=[],
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
            final_state.root_exec_time = bench_data.root_exec_time
            next_state, next_obs = self.reset()

        return next_obs, reward, done, next_state, final_state

    def get_obs(self, state: OperationState):
        """Build the obervation vector for the input state.

        Args:
            state (OperationState): the input state.

        Returns:
            np.ndarray: observation vector of the state.
        """

        op_features_vector = build_op_features_vector(state.operation_features)

        action_history = state.actions.reshape(-1)
        action_mask = state.actions_mask

        if state.operation_type == 'matmul':
            operation_type_int = 0
        elif 'conv_2d' in state.operation_type:
            operation_type_int = 1
        elif state.operation_type == 'pooling':
            operation_type_int = 2
        elif state.operation_type == 'add':
            operation_type_int = 3
        elif state.operation_type == 'generic':
            operation_type_int = 4

        operation_type_int_arr = np.array([operation_type_int])

        obs = np.concatenate((
            # The input of the policy network:
            operation_type_int_arr,  # 1
            op_features_vector,      # MAX_NUM_LOOPS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM*MAX_NUM_STORES_LOADS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM + 5
            action_history,  # MAX_NUM_LOOPS*3*CONFIG["truncate"]

            # The action mask:
            action_mask     # 5 + MAX_NUM_LOOPS + MAX_NUM_LOOPS + (MAX_NUM_LOOPS-1) + (MAX_NUM_LOOPS-2) + (MAX_NUM_LOOPS-3)
        ))

        # Normalize the upper bounds of the loops
        obs[1:cfg.max_num_loops + 1] = obs[1:cfg.max_num_loops + 1] / 100

        return obs

    def initialize_action_mask(self, num_loops: int, operation_type: str):
        """Initialize the action mask for a specified number of loops and operation type.

        Notes:
            Action mask (5 + L + L + (L-1) + (L-2) + (L-3) ):
                Transformations: end, TP, T, Interchange
                TP: L loops
                T : L loops
                Interchange: 2-consecutive interchanges: L - 1
                        : 3-consecutive interchanges: L - 2
                        : 4-consecutive interchanges: L - 3
                Interchange: 3L - 6

            action_mask[:5] = [end, TP, T, I, Img2Col]

        Args:
            num_loops (int): The number of loops in the operation.
            operation_type (str): The type of the operation.

        Returns:
            np.ndarray: The initialized action mask.
        """
        L = cfg.max_num_loops

        TP_BEGIN = 5
        T_BEGIN = TP_BEGIN + L
        I_BEGIN_2C = T_BEGIN + L
        I_BEGIN_3C = I_BEGIN_2C + (L - 1)
        I_BEGIN_4C = I_BEGIN_3C + (L - 2)

        action_mask = np.ones((5 + L + L + 3 * L - 6), dtype=np.bool_)
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

    def update_action_mask(self, state: OperationState, transformation: str, num_loops: int):
        """Update the action mask based on the transformation applied.

        Notes:
            actions_mask: (NUM_TRANSFORMATIONS + L + L + (L-1) + (L-2) + (L-3) )
            action_mask[:NUM_TRANSFORMATIONS] = [end, TP, T, I, Img2Col]

        Args:
            state (OperationState): The current state of the environment.
            transformation (str): The transformation applied.
            num_loops (int): The number of loops in the operation.

        Returns:
            np.ndarray: The updated action mask.
        """

        L = cfg.max_num_loops

        TP_BEGIN = cfg.num_transformations
        T_BEGIN = TP_BEGIN + L
        I_BEGIN_2C = T_BEGIN + L
        # I_BEGIN_3C = I_BEGIN_2C + (L-1)
        # I_BEGIN_4C = I_BEGIN_3C + (L-2)

        actions_mask = state.actions_mask

        if transformation == 'img2col':
            actions_mask[:TP_BEGIN] = [False, True, False, False, False]

        if state.operation_type == "pooling" or state.operation_type == "conv_2d":
            if transformation == 'parallelization':
                actions_mask[:TP_BEGIN] = [True, False, False, False, False]
            if transformation == 'tiling':
                actions_mask[:TP_BEGIN] = [True, False, False, False, False]

        elif state.operation_type == "conv_2d+img2col":
            if transformation == 'parallelization':
                actions_mask[:TP_BEGIN] = [True, False, False, False, False]

        elif state.operation_type == "matmul" or state.operation_type == "add":
            if transformation == 'parallelization':
                actions_mask[:TP_BEGIN] = [True, False, False, False, False]
            if transformation == 'tiling':
                actions_mask[:TP_BEGIN] = [True, False, True, True, False]
            if transformation == 'interchange':
                actions_mask[:TP_BEGIN] = [True, False, False, True, False]

        elif state.operation_type == "generic":
            if transformation == 'parallelization':
                actions_mask[:TP_BEGIN] = [True, False, False, False, False]
            if transformation == 'interchange':
                # NOTE: actions_mask[:NUM_TRANSFORMATIONS] = [True, False, True, True, False]
                actions_mask[:TP_BEGIN] = [True, True, True, True, False]
            if transformation == 'tiling':
                # NOTE: actions_mask[:NUM_TRANSFORMATIONS] = [True, False, False, False, False]
                actions_mask[:TP_BEGIN] = [True, True, True, True, False]

        else:
            raise ValueError("operation_type must be in [pooling, conv_2d, conv_2d+img2col, matmul, add, generic]")

        if num_loops == 1:
            actions_mask[3] = False
            actions_mask[I_BEGIN_2C] = True

        return actions_mask

    def update_action_history(self, state: OperationState, transformation: str, parameters: list[int]):
        """Update the action history based on the transformation applied.

        Args:
            state (OperationState): The current state of the environment.
            transformation (str): The transformation applied.
            parameters (list[int]): The parameters of the transformation.

        Returns:
            np.ndarray: The updated action history.
        """
        # actions.shape: (L, 3, truncate)
        # parallelization, tiling, interchange

        num_loops = len(state.operation_features.nested_loops)
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
        """Get all the possible interchanges for `num_loops`

        Args:
            num_loops (int): The number of loops in the operation.

        Returns:
            list[tuple]: The list of all possible interchanges.
        """

        interchanges = []
        for c in [1, 2, 3]:
            level_interchanges = []
            for _ in range(cfg.max_num_loops - c):
                level_interchanges.append(tuple(range(num_loops)))
            for i in range(num_loops - c):
                params = list(range(num_loops))
                params[i], params[i + c] = params[i + c], params[i]
                level_interchanges[i] = tuple(params)
            interchanges += level_interchanges
        return interchanges

    def sorted_divisors(self, n: int, num_candidates: int):
        """Get the divisors of `n` that are supperior or equal to 2

        Args:
            n (int): The upper bound.
            num_candidates (int): The number of candidates to get.

        Returns:
            list[int]: The sorted divisors.
        """

        divisors = []
        i = 1
        while i <= n and len(divisors) < num_candidates:
            if n % i == 0:
                divisors.append(i)
            i *= 2
        return sorted(divisors)

    def get_tiling_candidates(self, n: int, num_candidates: int, iterator_type: Literal['parallel', 'reduction'] = 'parallel'):
        """Get `num_candidates` candidate tiling size for upper bound `n`

        Args:
            n (int): The upper bound.
            num_candidates (int): The number of candidates to get.
            iterator_type (Literal['parallel', 'reduction']): The iterator type. Defaults to 'parallel'.

        Returns:
            list[int]: The tiling candidates.
        """

        # If upperbound equal 1, we only have candidates of 1
        if n == 1:
            return [1] * num_candidates

        # If the data format is json and the iterator type is reduction, we don't do tiling
        # TODO: the condition has to change because it's not related to the data format, it's related to a non thread safe tiling problem
        # so we skip it to not let it happen for now
        if cfg.data_format == 'json' and iterator_type == 'reduction':
            return [0] * num_candidates

        # We take the divisors of the upperbound `n`
        div = self.sorted_divisors(n, num_candidates)

        if len(div) < num_candidates:  # If we don't have enough unique divisors, we fill the rest of the candidates with the last dividor
            res = div + div[-1:] * (num_candidates - len(div))
        else:
            res = div[:num_candidates]
        return res

    def last_tiling(self, history: list[tuple[str, list[int]]]):
        """Get the last tiling from the action history

        Args:
            history (list[tuple[str, list[int]]]): The action history.

        Returns:
            Optional[list[int]]: The last tiling parameters if any.
        """
        for transformation, parameters in history[::-1]:
            if transformation in ['tiling', 'parallelization']:
                return parameters
        return None

    def process_action(self, raw_action: tuple[str, list[int]], state: OperationState):
        """Get the (transformation, parameters) from the `raw_action`.

        Args:
            raw_action (tuple[str, list[int]]): The raw action.

        Returns:
            tuple[str, list[int]]: The transformation and its parameters.
        """

        op_features = state.operation_features
        num_loops = len(op_features.nested_loops)
        action_name, parameter = raw_action

        # Sellect the tiling candidates for each loop
        if action_name in ['tiling', 'parallelization']:
            # Get loop upper bounds
            candidates = [
                [0] + self.get_tiling_candidates(loop.upper_bound, num_candidates=cfg.num_tile_sizes, iterator_type=loop.iterator_type)
                for loop in op_features.nested_loops
            ]

        if action_name == 'interchange':
            candidates = self.get_interchange_actions(num_loops)
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

            last_tiling_parameters = self.last_tiling(state.transformation_history)
            if last_tiling_parameters is not None:
                # TODO: Find out why this was added
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

    def speedup_reward(self, new: float, old: float, a: int = 10):
        """Get the reward based on the speedup.

        Args:
            new (float): The new execution time.
            old (float): The old execution time.
            a (int): The base of the logarithm. Defaults to 10.

        Returns:
            float: The calculated reward.
        """

        # if old >= new:
        #     reward = old/new - 1
        # else: # old < new
        #     reward = - new/old + 1

        # reward = math.log(old / new) / math.log(a)
        reward = math.log(old / new, a)

        return reward


class ParallelEnv:
    """Parallel environment for training the reinforcement learning agent."""

    num_env: int
    """number of environments."""
    envs: list[Env]
    """list of environments."""

    def __init__(self, num_env: int = 1, reset_repeat: int = 1, step_repeat: int = 1):
        """Initialize parallel environments.

        Args:
            num_env (int): number of environments. Defaults to 1.
            reset_repeat (int): The number of times to repeat the reset function. Defaults to 1.
            step_repeat (int): The number of times to repeat the step function. Defaults to 1.
        """
        self.num_env = num_env
        self.envs = [
            Env(
                reset_repeat=reset_repeat,
                step_repeat=step_repeat
            ) for i in range(num_env)
        ]

    def reset(self, idx: Optional[int] = None):
        """Reset the environments.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            list[OperationState]: The initial states of the environments.
        """
        states: list[OperationState] = []
        observations: list[torch.Tensor] = []
        for i in range(self.num_env):
            state, obs = self.envs[i].reset(idx=idx)
            states.append(state)
            observations.append(obs)
        return states, observations

    def step(self, states: list[OperationState], actions: list[tuple[str, list[int]]]) -> tuple[list[np.ndarray], list[float], list[bool], list[OperationState], list[Optional[OperationState]]]:
        """Take a step in the environments.

        Args:
            states (list[OperationState]): The current states of the environments.
            actions (list[tuple[str, list[int]]]): The raw actions taken by the agents.

        Returns:
            list[np.ndarray]: The observation vectors of the next states.
            list[float]: The rewards of the actions.
            list[bool]: Whether the episodes are done.
            list[OperationState]: The next states of the environments.
            list[Optional[OperationState]]: The final states of the environments if the episodes are done.
        """
        batch_next_obs: list[torch.Tensor] = []
        batch_reward: list[float] = []
        batch_done: list[bool] = []
        batch_next_state: list[OperationState] = []
        batch_final_state: list[Optional[OperationState]] = []

        for i, (state, action) in enumerate(zip(states, actions)):
            next_obs, reward, done, next_state, final_state = self.envs[i].step(state, action)
            batch_next_obs.append(next_obs)
            batch_reward.append(reward)
            batch_done.append(done)
            batch_next_state.append(next_state)
            batch_final_state.append(final_state)

        return batch_next_obs, batch_reward, batch_done, batch_next_state, batch_final_state
