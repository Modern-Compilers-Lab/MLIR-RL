from rl_autoschedular import config as cfg
from rl_autoschedular.state import (
    OperationState, BenchmarkFeatures, OperationFeatures,
    OperationType
)
from typing import Optional, Union, Literal
from rl_autoschedular.observation import (
    extract_bench_features_from_file,
    extract_bench_features_from_code,
    build_op_features_vector,
    update_operation_features,
    get_up_to_date_operation_features,
)
from rl_autoschedular.transforms import (
    apply_transformation,
    apply_conv2d_decomposition,
    is_vectorizable
)
from rl_autoschedular.evaluation import evaluate_code
from utils.log import print_error, print_success
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
    tmp_file: str
    """The temporary file to store the intermediate representations."""
    log_schedule: bool
    """Flag to log the schedule of the operations."""
    inference_env: bool
    """Flag to indicate if the environment is used for inference or training."""

    def __init__(self, tmp_file: Optional[str] = None, log_schedule: bool = False, inference_env: bool = False):
        """Initialize the environment.

        Args:
            tmp_file (Optional[str]): The temporary file to store the intermediate representations. Defaults to None.
        """
        # Generate a random file to be used in order to apply the transformations and evaluate the code
        # This is done in order to enable having multiple experiments at the same time, by letting each
        # experiment use a separate unique file to read and write intermediate representations
        self.log_schedule = log_schedule
        self.inference_env = inference_env
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
                benchmarks_json: dict[str, int] = json.load(file)
            # Build benchmark features
            for bench_name, root_exec_time in tqdm(benchmarks_json.items(), desc="Extracting benchmark features", unit="bench"):
                bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
                benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)
                self.benchmarks_data.append(benchmark_data)
        else:
            # Load operations data from json file
            with open(cfg.json_file, "r") as file:
                json_data = json.load(file)
            operation_filter = [
                'linalg.matmul',
                'linalg.conv_2d',
                # 'pooling',
                # 'linalg.generic',
                'linalg.add',
            ]
            json_data = {op: details for op, details in json_data.items() if any([s in op for s in operation_filter])}
            json_data: list[tuple[str, dict]] = [(details['operation'], details) for _, details in json_data.items()]

            # Get the AST of the MLIR code and give a tag to each linalg operation
            # The last operation represents the operations that we want to optimize (the first operations are just linalg.fills)
            for bench_name, details in tqdm(json_data, desc="Extracting benchmark features", unit="bench"):
                # Get full MLIR code and execution time
                code = details["transform_wrapped_operation"]
                root_exec_time = details["execution_time"]
                # Build benchmark features
                benchmark_data = extract_bench_features_from_code(bench_name, code, int(root_exec_time))
                if cfg.optimization_mode == "last":
                    last_op_tag = benchmark_data.operation_tags[-1]
                    benchmark_data.operation_tags = [last_op_tag]
                    benchmark_data.operations = {last_op_tag: benchmark_data.operations[last_op_tag]}
                    assert any([s in benchmark_data.operations[last_op_tag].raw_operation for s in operation_filter])
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
            print_error("Transformation Failed:", transformation, parameters)
            reward = self.__action_reward(trans_succeeded)
            next_state, next_obs, bench_done = self.__get_next_op_state(next_state)
            return next_state, next_obs, reward, True, 1.0, bench_done

        # Register the new code (transformation succeeded)
        next_state.transformed_code = new_transformed_code

        # Update the state infos to reflect the transformation
        self.__update_state_infos(next_state, transformation, parameters)

        # Evaluate the produced code and move on to another operation, if:
        # - The transformation is no_transformation or vectorization
        # - Maximum number of steps is reached
        if transformation in ['no_transformation', 'vectorization'] or next_state.step_count == cfg.truncate:
            try:
                new_exec_time, exec_succeeded = evaluate_code(next_state, bench_data)
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
            reward = self.__action_reward(trans_succeeded, exec_succeeded, new_exec_time, next_state.exec_time)
            speedup = (bench_data.root_exec_time / new_exec_time) if new_exec_time is not None else 1.0

            # Save the schedule in case we need to log it afterwards
            if self.log_schedule:
                state_history = next_state.transformation_history.copy()

            next_state, next_obs, bench_done = self.__get_next_op_state(next_state, new_exec_time)

            # Log the schedule if the benchmark is done
            if self.log_schedule and bench_done:
                print_success(f"Schedule: {state_history} - {speedup}")

            return next_state, next_obs, reward, True, speedup, bench_done

        # If the episode is not done, return the updated state with a reward of 0
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
        ])

    def __get_interchange_perm_vector(self, state: OperationState) -> np.ndarray:
        """Convert the interchange permutation to a vector.

        Args:
            state (OperationState): The current state.

        Returns:
            np.ndarray: The vector representation of the interchange permutation.
        """
        next_loop = len(state.interchange_permutation)
        assert next_loop < len(state.operation_features.nested_loops)
        perm_vector = np.zeros(cfg.max_num_loops, dtype=np.bool_)
        perm_vector[next_loop] = 1
        return perm_vector

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
        action_mask = self.__init_action_mask(operation_features)

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
            action_mask=action_mask,
            step_count=0,
            exec_time=bench_data.root_exec_time,
            transformation_history=[],
            interchange_permutation=[],
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

        # Reset to another benchmark if the current benchmark is done (reached first operation)
        if operation_idx == 0:
            return *self.reset(), True

        # Keep new code and execution time only if the new code is valid
        keep_new = new_exec_time is not None

        # Get the last validated history
        last_op_idx = state.last_op_history_index()
        if last_op_idx is None:
            # It means there are no loose transformations (without 'done' at the end)
            validated_history = state.transformation_history
        else:
            # Get all transformations up until the last 'done'
            validated_history = state.transformation_history[:last_op_idx]

        # Build a new state that points to the next operation
        new_op_tag = bench_data.operation_tags[operation_idx - 1]
        new_op_features = bench_data.operations[new_op_tag]
        new_op_type = self.__get_operation_type(new_op_features.raw_operation)
        new_action_mask = self.__init_action_mask(new_op_features)
        new_actions = self.__init_action_history()
        next_state = OperationState(
            bench_name=state.bench_name,
            operation_tag=new_op_tag,  # New operation tag
            operation_type=new_op_type,  # New operation type
            operation_features=new_op_features,  # New operation features
            validated_code=state.transformed_code if keep_new else state.validated_code,  # New code if keep new. Same code if not.
            transformed_code=state.transformed_code if keep_new else state.validated_code,  # Same code if keep new. Old code if not.
            actions=new_actions,  # Empty actions history
            action_mask=new_action_mask,  # New action mask
            step_count=0,  # Reset step count
            exec_time=new_exec_time if keep_new else state.exec_time,  # New execution time if keep new. Same execution time if not.
            transformation_history=state.transformation_history + [('done', [operation_idx])] if keep_new else validated_history,  # Same history (with done) if keep new. Validated history if not.
            interchange_permutation=[],  # Reset interchange permutation
            tmp_file=self.tmp_file,
        )

        return next_state, self.__get_obs(next_state), False

    def __init_action_mask(self, operation_features: OperationFeatures) -> np.ndarray:
        """Initialize the action mask.

        Notes:
            Action mask (NUM_TRANSFORMATIONS + L + L + interchange_mask):
                Transformations: no_transform, TP, T, I, vect, img2col
                TP: L loops
                T : L loops
                interchange_mask: 3 * L - 6 | L | 0

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
        I_BEGIN_1C = T_BEGIN + L
        I_BEGIN_2C = I_BEGIN_1C + L - 1
        I_BEGIN_3C = I_BEGIN_2C + L - 2

        match cfg.interchange_mode:
            case 'enumerate':
                interchange_mask = 3 * L - 6
            case 'pointers':
                interchange_mask = L
            case 'continuous':
                interchange_mask = 0

        action_mask = np.ones((cfg.num_transformations + 2 * L + interchange_mask), dtype=bool)
        if operation_type == 'conv_2d':
            action_mask[:cfg.num_transformations] = [False, False, False, False, False, True]
        else:
            action_mask[:cfg.num_transformations] = cfg.init_action_mask
        action_mask[TP_BEGIN + num_loops:T_BEGIN] = False
        action_mask[T_BEGIN + num_loops:I_BEGIN_1C] = False

        if cfg.interchange_mode == 'enumerate':
            action_mask[I_BEGIN_1C + max(num_loops - 1, 0):I_BEGIN_2C] = False
            action_mask[I_BEGIN_2C + max(num_loops - 2, 0):I_BEGIN_3C] = False
            action_mask[I_BEGIN_3C + max(num_loops - 3, 0):] = False
        elif cfg.interchange_mode == 'pointers':
            action_mask[I_BEGIN_1C + num_loops:] = False

        if num_loops == 1 and cfg.interchange_mode == 'enumerate':
            # If we have only one loop -> Allow the first candidate which will be the identity permutation
            action_mask[I_BEGIN_1C] = True

        action_mask = self.__ensure_feasible_vectorization(action_mask, operation_features)

        return action_mask

    def __update_action_mask(self, action_mask: np.ndarray, transformation: str, parameters: list[int], num_loops: int) -> np.ndarray:
        """Update the action mask based on the transformation applied.

        Args:
            state (OperationState): The current state of the environment.
            transformation (str): The transformation applied.
            parameters (list[int]): The parameters of the transformation.
            num_loops (int): The number of loops in the operation.

        Returns:
            np.ndarray: The updated action mask.
        """
        new_action_mask = action_mask.copy()

        N = cfg.num_transformations
        L = cfg.max_num_loops
        I_BEGIN = N + 2 * L

        if cfg.interchange_mode == 'pointers':
            # Reset pointer masking
            new_action_mask[I_BEGIN:I_BEGIN + num_loops] = True

        match transformation:
            case 'img2col':
                new_action_mask[:N] = cfg.init_action_mask
            case 'parallelization':
                new_action_mask[:N] = [not cfg.force_vector, False, False, False, True, False]
            case 'tiling':
                new_action_mask[:N] = [False, False, False, True, False, False]
            case 'interchange':
                if len(parameters) > 0 and len(parameters) < num_loops:
                    # In case of incomplete interchange, prevent any other action, and prevent repeating a loop
                    new_action_mask[:N] = [False, False, False, True, False, False]
                    for param in parameters:
                        new_action_mask[I_BEGIN + param] = False
                else:
                    new_action_mask[:N] = [False, True, False, False, False, False]

        if num_loops == 1 and cfg.interchange_mode == 'enumerate':
            # If we have only one loop -> Allow the first candidate which will be the identity permutation
            new_action_mask[I_BEGIN] = True

        return new_action_mask

    def __ensure_feasible_vectorization(self, action_mask: np.ndarray, operation_features: OperationFeatures) -> np.ndarray:
        """Automatically disable unfeasible vectorization if we should to.

        Args:
            action_mask (np.ndarray): The action mask.
            operation_features (OperationFeatures): The operation features.

        Returns:
            np.ndarray: The updated action mask.
        """
        new_action_mask = action_mask.copy()

        # Nothing to do, If:
        # - Vectorization is already disabled
        # - It's a training env and we want to punish the agent for unfeasible vectorization
        if (not new_action_mask[4]) or (not self.inference_env and cfg.punish_vector):
            return new_action_mask

        # Check if vectorization is feasible
        vectorizable = is_vectorizable(operation_features)

        # If vectorization is feasible, nothing to do
        if vectorizable:
            return new_action_mask

        # Otherwise, turn vectorization into no_transformation
        new_action_mask[4] = False
        new_action_mask[0] = True

        return new_action_mask

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
        if cfg.reverse_history:
            return np.zeros((cfg.max_num_loops, 3, cfg.truncate))
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
        if cfg.reverse_history:
            assert state.step_count < state.actions.shape[2]
        else:
            assert state.step_count < state.actions.shape[0]

        transformation_indices = {
            'parallelization': 0,
            'tiling': 1,
            'interchange': 2
        }
        transformation_index = transformation_indices[transformation]
        for loop_index in range(num_loops):
            if cfg.reverse_history:
                new_actions[loop_index, transformation_index, state.step_count] = parameters[loop_index]
            else:
                new_actions[state.step_count, transformation_index, loop_index] = parameters[loop_index]

        return new_actions

    def __get_obs(self, state: OperationState) -> torch.Tensor:
        """Build the obervation vector for the input state.

        Args:
            state (OperationState): the input state.

        Returns:
            np.ndarray: observation vector of the state.
        """
        op_type_vector = self.__get_op_type_vector(state.operation_type)
        op_features_vector = build_op_features_vector(state.operation_features)
        if cfg.interchange_mode == 'pointers':
            interchange_perm_vector = self.__get_interchange_perm_vector(state)
            op_features_vector = np.concatenate((op_features_vector, interchange_perm_vector))

        action_history = state.actions.reshape(-1)
        action_mask = state.action_mask

        obs = np.concatenate((
            # The input of the policy network:
            op_type_vector,  # 5
            op_features_vector,  # MAX_NUM_LOOPS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM*MAX_NUM_STORES_LOADS + MAX_NUM_LOOPS*MAX_NUM_LOAD_STORE_DIM + 5 [+ MAX_NUM_LOOPS]
            action_history,  # truncate*3*MAX_NUM_LOOPS

            # The action mask:
            action_mask  # 5 + MAX_NUM_LOOPS + MAX_NUM_LOOPS
        ))

        # Normalize the upper bounds of the loops
        if cfg.normalize_bounds:
            obs[5:cfg.max_num_loops + 5] = obs[5:cfg.max_num_loops + 5] / 100

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
            # Get updated features
            operation_features = get_up_to_date_operation_features(state)

            # Get loop upper bounds
            candidates = [
                [0] + self.__get_tiling_candidates(loop.upper_bound, loop.iterator_type, action_name == 'parallelization')
                for loop in operation_features.nested_loops
            ]

        if action_name == 'interchange':
            match cfg.interchange_mode:
                case 'enumerate':
                    candidates = self.__get_interchange_candidates(num_loops)
                    parameters = candidates[parameter]
                    assert len(parameters) == num_loops
                case 'pointers':
                    assert parameter not in state.interchange_permutation, f"Loop {parameter} is already in the interchange permutation: {state.interchange_permutation}"
                    assert len(state.interchange_permutation) < num_loops
                    parameters = state.interchange_permutation + [parameter]
                case 'continuous':
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

    def __get_tiling_candidates(self, n: int, iterator: Literal['parallel', 'reduction'], for_parallelization: bool) -> list[int]:
        """Get the tiling candidates for a given loop.

        Args:
            n (int): The loop upper bound.
            iterator_type (Literal['parallel', 'reduction']): The type of the loop iterator.
            for_parallelization (bool): A flag indicating if the tiling is for parallelization.

        Returns:
            list[int]: The tiling candidates.
        """
        num_candidates = cfg.num_tile_sizes

        # If there is only one or no iteration, then, no tiling is possible
        if n <= 1 or (iterator == 'reduction' and for_parallelization):
            return [0] * num_candidates

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

    def __get_interchange_candidates(self, num_loops: int) -> list[list[int]]:
        """Get all 1c 2c 3c possible interchanges for `num_loops`

        Args:
            num_loops (int): The number of loops in the operation.

        Returns:
            list[tuple]: The list of all possible interchanges.
        """

        interchanges = []
        for c in [1, 2, 3]:
            level_interchanges = []
            for _ in range(cfg.max_num_loops - c):
                level_interchanges.append(list(range(num_loops)))
            for i in range(num_loops - c):
                params = list(range(num_loops))
                params[i], params[i + c] = params[i + c], params[i]
                level_interchanges[i] = params
            interchanges += level_interchanges
        return interchanges

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
        if x >= math.factorial(n):
            x = math.factorial(n) - 1

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
        num_loops = len(state.operation_features.nested_loops)
        transformed_code: Optional[str] = None
        match transformation:
            case 'parallelization' | 'tiling' | 'img2col':
                # Apply the transformation and get the new code
                try:
                    transformed_code = apply_transformation(
                        state=state,
                        code=state.transformed_code,
                        transformation=transformation,
                        parameters=parameters,
                        in_inference=self.inference_env,
                    )
                except Exception as e:
                    print_error(f"Error while applying the transformation: {e}")
                    return '', False

            case 'interchange':
                if len(parameters) == num_loops:
                    # We apply interchange only when the permutation is complete
                    try:
                        transformed_code = apply_transformation(
                            state=state,
                            code=state.transformed_code,
                            transformation=transformation,
                            parameters=parameters,
                            in_inference=self.inference_env,
                        )
                    except Exception as e:
                        print_error(f"Error while applying the transformation: {e}")
                        return '', False
                else:
                    # We keep the same code as previously if the permutation is not complete
                    transformed_code = state.transformed_code

            case 'no_transformation' | 'vectorization':
                # For convolution, before vectorization, we need to first apply another tiling in order to decompose it to 1d convolution
                if (state.operation_type == 'conv_2d'):
                    second_interchange_parameters = None
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
                    if second_interchange_parameters is not None:
                        try:
                            transformed_code = apply_transformation(
                                state=state,
                                code=state.transformed_code,
                                transformation='tiling',
                                parameters=second_interchange_parameters,
                                in_inference=self.inference_env,
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
                        transformed_code = apply_transformation(
                            state=state,
                            code=state.transformed_code,
                            transformation=transformation,
                            parameters=parameters,
                            in_inference=self.inference_env,
                        )
                    except Exception as e:
                        print_error(f"Error while applying the transformation: {e}")
                        return '', False

            case _:
                raise ValueError(f"Invalid transformation: {transformation}")

        return transformed_code, bool(transformed_code)

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
            return -5.0

        assert exec_succeeded is not None
        if not exec_succeeded:
            return -20.0

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

        # if old < new * 2:
        #     return math.log(old / (new * 2))
        # else:
        #     return old / (new * 2) - 1
        return math.log10(old / new)

    def __update_state_infos(self, state: OperationState, transformation: str, parameters: list[int]):
        """Update state infos after applying a transformation.

        Notes: Updated fields are:
            - operation_features (to reflect the transformation)
            - operation_type (only in case of conv2d+img2col)
            - actions (to register the transformation for the next input)
            - transformation_history
            - action_mask
            - step _count

        Args:
            state (OperationState): The current state.
            transformation (str): The transformation applied.
            parameters (list[int]): The transformation parameters.

        Returns:
            OperationState: The updated state.
        """
        num_loops = len(state.operation_features.nested_loops)

        # Update the action mask
        state.action_mask = self.__update_action_mask(state.action_mask, transformation, parameters, num_loops)

        # If interchange permutation is not complete
        #  -> Record it in the state, and return
        if transformation == 'interchange' and len(parameters) < num_loops:
            state.interchange_permutation = parameters
            return

        # Erase saved interchange permutation (Not needed anymore)
        state.interchange_permutation = []

        # Get updated operation features if:
        # - The transformation is img2col (necessary)
        # - The config says so
        if cfg.update_op_features or transformation == 'img2col':
            state.operation_features = update_operation_features(state, transformation, parameters)

        # Register the action in the history
        state.actions = self.__update_action_history(state, transformation, parameters)
        state.transformation_history.append((transformation, parameters))

        # Verify if vectorization is feasible after the transformation
        up_to_date_op_features = get_up_to_date_operation_features(state)
        state.action_mask = self.__ensure_feasible_vectorization(state.action_mask, up_to_date_op_features)

        # Update the step count
        state.step_count += 1
