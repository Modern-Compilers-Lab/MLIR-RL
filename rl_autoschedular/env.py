from rl_autoschedular import config as cfg
from rl_autoschedular.state import (
    OperationState, BenchmarkFeatures, OperationFeatures,
    OperationType
)
from typing import Optional, Union, Literal
from rl_autoschedular.observation import (
    extract_bench_features_from_file,
    extract_bench_features_from_code,
    extract_op_features_from_affine_code,
    build_op_features_vector,
)
from rl_autoschedular.transforms import (
    apply_transformation_with_timeout,
    get_ops_by_tags,
    apply_conv2d_decomposition
)
from rl_autoschedular.evaluation import evaluate_code_with_timeout
from utils.log import print_error, print_info
from tqdm import tqdm
import numpy as np
import torch
import random
import string
import json
import os
import math


class Env:
    """RL Environment class"""

    benchmarks_data: list[BenchmarkFeatures]
    """Lists for each benchmark the benchmark's name and its features."""
    is_inference_env: bool
    """Flag to indicate if the environment is used for inference."""
    tmp_file: str
    """The temporary file to store the intermediate representations."""

    def __init__(self, is_inference_env: bool = False, tmp_file: Optional[str] = None):
        """Initialize the environment.

        Args:
            reset_repeat (int): The number of times to repeat the reset function. Defaults to 1.
            step_repeat (int): The number of times to repeat the step function. Defaults to 1.
            tmp_file (Optional[str]): The temporary file to store the intermediate representations. Defaults to None.
        """
        self.is_inference_env = is_inference_env

        # Generate a random file to be used in order to apply the transformations and evaluate the code
        # This is done in order to enable having multiple experiments at the same time, by letting each
        # experiment use a separate unique file to read and write intermediate representations
        if tmp_file is None:
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            tmp_file = f"tmp/{random_str}.mlir"
        with open(tmp_file, "w") as file:
            file.write("")
        self.tmp_file = tmp_file

        # Get benchmarks data
        # TODO: Try to unify this process
        self.benchmarks_data = []
        if cfg.data_format == "mlir":
            # Load execution times from json file
            with open(cfg.json_file, "r") as file:
                benchmarks_json: dict[str, float] = json.load(file)
            # Build benchmark features
            for bench_name, root_exec_time in benchmarks_json.items():
                bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, int(root_exec_time * 10**9))
                self.benchmarks_data.append(benchmark_data)
        else:
            # Load operations data from json file
            with open(cfg.json_file, "r") as file:
                json_data = json.load(file)
            operation_filter = [
                'linalg.matmul',
                'linalg.conv_2d',
                # 'pooling',
                'linalg.generic',
                'linalg.add',
            ]
            json_data = {op: details for op, details in json_data.items() if any([s in op for s in operation_filter])}
            json_data = [(details['operation'], details) for _, details in json_data.items()]

            # Get the AST of the MLIR code and give a tag to each linalg operation
            # The last operation represents the operations that we want to optimize (the first operations are just linalg.fills)
            for i in tqdm(range(len(json_data))):
                # Get full MLIR code and execution time
                code = json_data[i][1]["transform_wrapped_operation"]
                root_exec_time = json_data[i][1]["execution_time"]
                # Build benchmark features
                bench_name = f"bench_{i}"
                benchmark_data = extract_bench_features_from_code(bench_name, code, int(root_exec_time))
                self.benchmarks_data.append(benchmark_data)

    def reset(self, idx: Optional[int] = None) -> tuple[OperationState, torch.Tensor]:
        """Reset the environment.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            OperationState: The initial state of the environment.
            torch.Tensor: The observation vector of the initial state.
        """
        # Get the benchmark
        if idx is not None:
            self.bench_index = idx
        else:
            self.bench_index = random.randint(0, len(self.benchmarks_data) - 1)
        benchmark_data = self.benchmarks_data[self.bench_index]

        return self.__init_op_state(benchmark_data, -1)

    def step(self, state: OperationState, raw_action: tuple[str, Optional[Union[list[int], int]]]) -> tuple[OperationState, torch.Tensor, float, bool, Optional[float], bool]:
        """Take a step in the environment.

        Args:
            state (OperationState): The current state.
            raw_action (tuple[str, Optional[Union[list[int], int]]]): The raw action to take.

        Returns:
            OperationState: The new state.
            Tensor: The observation vector of the new state.
            float: The reward of the action.
            bool: A flag indicating if the episode is done.
            Optional[float]: The speedup (if the operation is executed successfully) for loggin purposes.
            bool: A flag indicating if the benchmark is done.
        """
        bench_data = self.benchmarks_data[self.bench_index]

        # Copy the current state to introduce the changes throughout the function
        next_state = state.copy()

        # Process the raw action
        transformation, parameters = self.__process_action(raw_action, next_state)

        # Attempt to apply the transformation to the code
        # - If the transformation fails: punish the agent, reset the code, and move on to another operation
        new_transformed_code, trans_succeeded = self.__handle_transformation(transformation, parameters, state)
        if not trans_succeeded:
            next_state, next_obs, bench_done = self.__get_next_op_state(next_state)
            reward = self.__action_reward(trans_succeeded)
            return next_state, next_obs, reward, True, None, bench_done

        # Register the new code and the new transformation (transformation succeeded)
        next_state.transformed_code = new_transformed_code
        next_state.transformation_history.append((transformation, parameters))

        # Evaluate the produced code and move on to another operation, if:
        # - The transformation is no_transformation or vectorization
        # - Maximum number of steps is reached
        if transformation in ['no_transformation', 'vectorization'] or next_state.step_count == (cfg.truncate - 1):
            try:
                new_exec_time, exec_succeeded = evaluate_code_with_timeout(next_state, bench_data)
                if isinstance(exec_succeeded, Exception):
                    raise exec_succeeded
                if not exec_succeeded or new_exec_time is None:
                    raise Exception("Execution failed")
            except Exception as e:
                print_error(f"Error while evaluating the code: {e}")
                exec_succeeded = False
                new_exec_time = None

            # Next state and reward will take into consideration whether execution succeeded or not
            # i.e: if execution failed: punish the agent, reset the code, and move on to another operation
            if self.is_inference_env:
                tmp_history = next_state.transformation_history.copy()
            next_state, next_obs, bench_done = self.__get_next_op_state(next_state, new_exec_time)
            if self.is_inference_env and bench_done:
                print_info(f"History: {tmp_history}")
            reward = self.__action_reward(trans_succeeded, exec_succeeded, new_exec_time, next_state.exec_time)
            speedup = bench_data.root_exec_time / new_exec_time if new_exec_time is not None else None
            return next_state, next_obs, reward, True, speedup, bench_done

        # Update the state infos
        self.__update_state_infos(next_state, transformation, parameters)

        return next_state, self.__get_obs(next_state), 0.0, False, None, False

    def __get_operation_type(self, raw_operation: str) -> OperationType:
        """Get the operation type from the raw operation string.

        Args:
            raw_operation (str): The raw operation string.

        Returns:
            str: The operation type.
        """
        if 'linalg.matmul' in raw_operation:
            return 'matmul'
        elif 'linalg.conv' in raw_operation:
            return 'conv_2d'
        elif 'pooling' in raw_operation:
            return 'pooling'
        elif 'linalg.add' in raw_operation:
            return 'add'
        elif 'linalg.generic' in raw_operation:
            return 'generic'
        else:
            raise ValueError(f"Unknown operation type: {raw_operation}")

    def __get_op_type_vector(self, op_type: OperationType):
        """Convert the operation type to an integer.

        Args:
            op_type (OperationType): The operation type.

        Returns:
            int: The integer representation of the operation type.
        """
        return np.array([
            op_type == 'generic',
            op_type == 'matmul',
            op_type == 'add',
            op_type == 'pooling',
            op_type == 'conv_2d',
            op_type == 'conv_2d+img2col',
        ])

    def __init_op_state(self, bench_data: BenchmarkFeatures, operation_idx: int) -> tuple[OperationState, torch.Tensor]:
        """Create a new operation state.

        Args:
            bench_idx (int): The benchmark index.
            operation_idx (int): The operation index.

        Returns:
            OperationState: The new operation state.
            torch.Tensor: The observation vector of the new operation state.
        """
        operation_tag = bench_data.operation_tags[operation_idx]
        operation_features = bench_data.operations[operation_tag]

        # Get operation type
        operation_type = self.__get_operation_type(operation_features.raw_operation)

        # Build action mask
        actions_mask = self.__init_action_mask(operation_features)

        # Create empty action history
        actions = self.__init_action_history()

        state = OperationState(
            bench_name=bench_data.bench_name,
            operation_tag=operation_tag,
            operation_type=operation_type,
            operation_features=operation_features.copy(),
            validated_code=bench_data.code,
            transformed_code=bench_data.code,
            actions=actions,
            actions_mask=actions_mask,
            step_count=0,
            exec_time=bench_data.root_exec_time,
            transformation_history=[],
            tmp_file=self.tmp_file,
        )

        return state, self.__get_obs(state)

    def __get_next_op_state(self, state: OperationState, new_exec_time: Optional[float] = None) -> tuple[OperationState, torch.Tensor, bool]:
        """Get the state that represents the next operation (can be from another benchmark).

        Args:
            state (OperationState): The current state.

        Returns:
            OperationState: The next state.
            torch.Tensor: The observation vector of the next state.
            bool: Flag indicating if the benchmark is done.
        """
        bench_data = self.benchmarks_data[self.bench_index]
        operation_idx = bench_data.operation_tags.index(state.operation_tag)

        # Reset to another benchmark if:
        # - the current benchmark is done (reached first operation)
        # - in inference mode and the new code is invalid (becuase optimizations for all operations is needed in evaluation)
        if operation_idx == 0 or (self.is_inference_env and new_exec_time is None):
            return *self.reset(), True

        # Keep new code and execution time if:
        # - in inference mode
        # - the new code is valid
        keep_new = self.is_inference_env and new_exec_time is not None

        # Build a new state that points to the next operation
        new_op_tag = bench_data.operation_tags[operation_idx - 1]
        new_op_features = bench_data.operations[new_op_tag]
        new_op_type = self.__get_operation_type(new_op_features.raw_operation)
        new_actions_mask = self.__init_action_mask(new_op_features)
        new_actions = self.__init_action_history()
        next_state = OperationState(
            bench_name=state.bench_name,
            operation_tag=new_op_tag,  # New operation tag
            operation_type=new_op_type,  # New operation type
            operation_features=new_op_features,  # New operation features
            validated_code=state.transformed_code if keep_new else state.validated_code,  # New code if keep new. Same code if not.
            transformed_code=state.transformed_code if keep_new else state.validated_code,  # Same code if keep new. Old code if not.
            actions=new_actions,  # Empty actions history
            actions_mask=new_actions_mask,  # New action mask
            step_count=0,  # Reset step count
            exec_time=new_exec_time if keep_new else state.exec_time,  # New execution time if keep new. Same execution time if not.
            transformation_history=state.transformation_history + [('done', [operation_idx])] if keep_new else [],  # Same history (with done) if keep new. Empty history if not.
            tmp_file=self.tmp_file,
        )

        return next_state, self.__get_obs(next_state), False

    def __init_action_mask(self, operation_features: OperationFeatures) -> np.ndarray:
        """Initialize the action mask.

        Notes:
            Action mask (NUM_TRANSFORMATIONS + L + L):
                Transformations: no_transform, TP, T, I, vect, img2col
                TP: L loops
                T : L loops

        Args:
            operation_features (OperationFeatures): The operation features.

        Returns:
            np.ndarray: The initialized action mask.
        """
        operation_type = self.__get_operation_type(operation_features.raw_operation)
        num_loops = len(operation_features.nested_loops)

        L = cfg.max_num_loops
        TP_BEGIN = cfg.num_transformations
        T_BEGIN = TP_BEGIN + L

        action_mask = np.ones((cfg.num_transformations + L + L), dtype=np.bool_)
        if operation_type == 'conv_2d':
            action_mask[:cfg.num_transformations] = [False, False, False, False, False, True]
        else:
            action_mask[:cfg.num_transformations] = [False, True, False, False, False, False]
        action_mask[TP_BEGIN + num_loops:T_BEGIN] = False
        action_mask[T_BEGIN + num_loops:] = False

        if num_loops == 1:
            # If we have only one loop -> Cancel the interchange
            action_mask[3] = False

        return action_mask

    def __update_action_mask(self, state: OperationState, transformation: str, parameters: list[int], num_loops: int) -> np.ndarray:
        """Update the action mask based on the transformation applied.

        Args:
            state (OperationState): The current state of the environment.
            transformation (str): The transformation applied.
            parameters (list[int]): The parameters of the transformation.
            num_loops (int): The number of loops in the operation.

        Returns:
            np.ndarray: The updated action mask.
        """
        TP_BEGIN = cfg.num_transformations

        new_actions_mask = state.actions_mask.copy()

        match transformation:
            case 'img2col':
                new_actions_mask[:TP_BEGIN] = [False, True, False, False, False, False]
            case 'parallelization':
                new_actions_mask[:TP_BEGIN] = [True, False, False, False, True, False]
            case 'tiling':
                new_actions_mask[:TP_BEGIN] = [True, False, True, True, True, False]
            case 'interchange':
                new_actions_mask[:TP_BEGIN] = [True, True, True, True, True, False]

        return new_actions_mask

    def __init_action_history(self) -> np.ndarray:
        """Initialize the action history.

        Notes:
            The action history is a 3D array with the shape (truncate, 3, MAX_NUM_LOOPS).
            Second index:
                0: parallelization
                1: tiling
                2: interchange

        Returns:
            np.ndarray: The initialized action history.
        """
        return np.zeros((cfg.truncate, 3, cfg.max_num_loops))

    def __update_action_history(self, state: OperationState, transformation: str, parameters: list[int]) -> np.ndarray:
        """Update the action history based on the transformation applied.

        Args:
            state (OperationState): The current state of the environment.
            transformation (str): The transformation applied.
            parameters (list[int]): The parameters of the transformation.

        Returns:
            np.ndarray: The updated action history.
        """
        new_actions = state.actions.copy()
        num_loops = len(state.operation_features.nested_loops)
        if len(parameters) < num_loops or transformation in ['no_transformation', 'vectorization', 'img2col']:
            return new_actions
        assert state.step_count < state.actions.shape[0]

        # actions[s, t, l] = the parameters of transformation `t` for loop `l` at step `s`
        transformation_indices = {
            'parallelization': 0,
            'tiling': 1,
            'interchange': 2
        }
        transformation_index = transformation_indices[transformation]
        for loop_index in range(num_loops):
            new_actions[state.step_count, transformation_index, loop_index] = parameters[loop_index]

        return new_actions

    def __get_obs(self, state: OperationState) -> torch.Tensor:
        """Build the obervation vector for the input state.

        Args:
            state (OperationState): the input state.

        Returns:
            np.ndarray: observation vector of the state.
        """

        op_features_vector = build_op_features_vector(state.operation_features)
        op_type_vector = self.__get_op_type_vector(state.operation_type)

        action_history = state.actions.reshape(-1)
        action_mask = state.actions_mask

        obs = np.concatenate((
            # The input of the policy network:
            op_type_vector,  # 6
            op_features_vector,  # MAX_NUM_LOOPS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM*MAX_NUM_STORES_LOADS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM + 5
            action_history,  # CONFIG["truncate"]*3*MAX_NUM_LOOPS

            # The action mask:
            action_mask  # 5 + MAX_NUM_LOOPS + MAX_NUM_LOOPS
        ))

        # Normalize the upper bounds of the loops
        # obs[5:cfg.max_num_loops + 1] = obs[5:cfg.max_num_loops + 1] / 100

        obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.unsqueeze(0)

        return obs

    def __process_action(self, raw_action: tuple[str, Optional[Union[list[int], int]]], state: OperationState) -> tuple[str, list[int]]:
        """Process the raw action.

        Args:
            raw_action (tuple[str, Optional[Union[list[int], int]]]): The raw action.
            state (OperationState): The current state.

        Returns:
            str: Action name.
            list[int]: Action parameters.
        """
        num_loops = len(state.operation_features.nested_loops)
        action_name, parameter = raw_action

        # Sellect the tiling candidates for each loop
        if action_name in ['tiling', 'parallelization']:
            # Get loop upper bounds
            candidates = [
                [0] + self.__get_tiling_candidates(loop.upper_bound, loop.iterator_type, action_name == 'parallelization')
                for loop in state.operation_features.nested_loops
            ]

        if action_name == 'interchange':
            parameters = self.__decode_interchange_parameter(parameter, num_loops)
            assert len(parameters) == num_loops
            return ('interchange', parameters)

        elif action_name == 'img2col':
            return ('img2col', [0])

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

            return ('tiling', tiling_parameters)

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

            return ('parallelization', parall_parameters)

        elif action_name == 'vectorization':
            return ('vectorization', [0])

        return ('no_transformation', [0])

    def __get_tiling_candidates(self, n: int, iterator_type: Literal['parallel', 'reduction'] = 'parallel', for_parallelization: bool = False) -> list[int]:
        """Get the tiling candidates for a given loop.

        Args:
            n (int): The loop upper bound.
            iterator_type (Literal['parallel', 'reduction']): The type of the loop iterator.
            for_parallelization (bool): A flag indicating if the tiling is for parallelization.

        Returns:
            list[int]: The tiling candidates.
        """
        num_candidates = cfg.num_tile_sizes

        # We can't parallelize a reduction loop
        if iterator_type == 'reduction' and for_parallelization:
            return [0] * num_candidates

        # If there is only one iteration, then the only possible tiling is 1
        if n == 1:
            return [1] * num_candidates

        # Get the factors of the loop upper bound
        factors = []
        f = 1
        while len(factors) < num_candidates:
            if f >= n:
                factors += [0] * (num_candidates - len(factors))
                break

            if n % f == 0:
                factors.append(f)
            f *= 2

        return factors

    def __decode_interchange_parameter(self, parameter: int, num_loops: int) -> list[int]:
        """Decode the interchange parameter to get the loop permutation.

        Args:
            parameter (int): The interchange parameter.
            num_loops (int): The number of loops in the operation.

        Returns:
            list[int]: The loop permutation.
        """
        x = parameter
        n = num_loops

        # Convert x to factorial number
        fact_x = '0'
        q = x
        d = 2
        while q > 0:
            r = q % d
            q = q // d
            fact_x = str(r) + fact_x
            d += 1

        # Ensure to get exactly n digits
        fact_x = fact_x.zfill(n)[-n:]

        # Decode factorial number following Lehmer code
        nl = list(map(int, fact_x))
        for i in range(len(nl) - 2, -1, -1):
            for j in range(i + 1, len(nl)):
                if nl[j] >= nl[i]:
                    nl[j] += 1

        return nl

    def __handle_transformation(self, transformation: str, parameters: list[int], state: OperationState) -> tuple[str, bool]:
        """Apply the transformation to the code along with some additional checks.

        Args:
            transformation (str): The transformation name.
            parameters (list[int]): The transformation parameters.
            state (OperationState): The current state.

        Returns:
            str: The transformed code.
            bool: A flag indicating if the transformation was successful.
        """
        transformed_code: Optional[str] = None
        match transformation:
            case 'parallelization' | 'tiling' | 'img2col' | 'interchange':
                # Apply the transformation and get the new code
                try:
                    transformed_code = apply_transformation_with_timeout(
                        state=state,
                        code=state.transformed_code,
                        transformation=transformation,
                        parameters=parameters,
                    )
                except Exception as e:
                    print_error(f"Error while applying the transformation: {e}")
                    return '', False

            case 'no_transformation' | 'vectorization':
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
                    try:
                        transformed_code = apply_transformation_with_timeout(
                            state=state,
                            code=state.transformed_code,
                            transformation='tiling',
                            parameters=second_interchange_parameters,
                        )

                        transformed_code = apply_conv2d_decomposition(transformed_code, state.operation_tag, self.tmp_file)
                    except Exception as e:
                        print_error(f"Error while applying the transformation: {e}")
                        return '', False

                if state.operation_type == 'pooling':
                    # Force no transformation on pooling operations
                    transformation = 'no_transformation'
                    transformed_code = state.transformed_code
                else:
                    # Otherwise apply the transformation and get the new code
                    try:
                        transformed_code = apply_transformation_with_timeout(
                            state=state,
                            code=state.transformed_code,
                            transformation=transformation,
                            parameters=parameters,
                        )
                    except Exception as e:
                        print_error(f"Error while applying the transformation: {e}")
                        return '', False

            case _:
                raise ValueError(f"Invalid transformation: {transformation}")

        return transformed_code, bool(transformed_code)

    def __update_operation_features(self, state: OperationState, transformation: str, parameters: list[int]) -> tuple[OperationFeatures, OperationType]:
        """Update the operation features after applying a transformation.

        Args:
            operation_features (OperationFeatures): The operation features.
            transformation (str): The transformation name.
            parameters (list[int]): The transformation parameters.

        Returns:
            OperationFeatures: The updated operation features.
            OperationType: The updated operation type.
        """
        new_operation_features = state.operation_features.copy()
        if transformation in ['no_transformation', 'vectorization']:
            return new_operation_features

        new_operation_type = state.operation_type
        match transformation:
            case 'parallelization' | 'tiling':
                for nested_loop, tile_size in zip(new_operation_features.nested_loops, parameters):
                    if tile_size == 0:
                        continue
                    nested_loop.upper_bound = tile_size
            case 'interchange':
                for i, j in enumerate(parameters):
                    new_operation_features.nested_loops[i] = state.operation_features.nested_loops[j]
            case 'img2col':
                # Get the matmul operation that now represents the convolution and wrap it in a funciton wrapper
                # to prepare it for the optimization in the next iterations
                prints = get_ops_by_tags(state.transformed_code, [state.operation_tag], self.tmp_file)
                raw_operation = list(prints.values())[0]
                new_operation_features = extract_op_features_from_affine_code(raw_operation, self.tmp_file)
                new_operation_type = 'conv_2d+img2col'
            case _:
                raise ValueError(f"Invalid transformation: {transformation}")

        return new_operation_features, new_operation_type

    def __action_reward(self, trans_succeeded: bool, exec_succeeded: Optional[bool] = None, new_exec_time: Optional[int] = None, old_exec_time: Optional[int] = None) -> float:
        """Get the reward of the action based on the transformation and execution results.

        Args:
            trans_succeeded (bool): A flag indicating if the transformation was successful.
            exec_succeeded (Optional[bool]): A flag indicating if the execution was successful. (required if trans succeeded)
            new_exec_time (Optional[float]): The execution time after transformation. (required if exec succeeded)
            old_exec_time (Optional[float]): The original execution time. (required if exec succeeded)

        Returns:
            float: The reward of the action.
        """
        if not trans_succeeded:
            return -10.0

        assert exec_succeeded is not None
        if not exec_succeeded:
            return -50.0

        assert new_exec_time is not None and old_exec_time is not None
        return self.__speedup_reward(new_exec_time, old_exec_time)

    def __speedup_reward(self, new: int, old: int) -> float:
        """Get the reward based on the speedup.

        Args:
            new (float): The new execution time.
            old (float): The old execution time.
            a (int): The base of the logarithm. Defaults to 10.

        Returns:
            float: The calculated reward.
        """

        if old < new * 2:
            return math.log(old / (new * 2))
        else:
            return old / (new * 2) - 1

    def __update_state_infos(self, state: OperationState, transformation: str, parameters: list[int]):
        """Update state infos after applying a transformation.

        Notes: Updated fields are:
            - operation_features (to reflect the transformation)
            - operation_type (only in case of conv2d+img2col)
            - actions (to register the transformation for the next input)
            - transformation_history
            - actions_mask
            - step _count

        Args:
            state (OperationState): The current state.
            transformation (str): The transformation applied.
            parameters (list[int]): The transformation parameters.

        Returns:
            OperationState: The updated state.
        """
        # Get updated operation features
        new_op_features, new_op_type = self.__update_operation_features(state, transformation, parameters)
        state.operation_features = new_op_features
        state.operation_type = new_op_type

        # Register the action in the history
        new_actions = self.__update_action_history(state, transformation, parameters)
        state.actions = new_actions

        # Update the action mask
        new_actions_mask = self.__update_action_mask(state, transformation, parameters, len(new_op_features.nested_loops))
        state.actions_mask = new_actions_mask

        # Update the step count
        state.step_count += 1
