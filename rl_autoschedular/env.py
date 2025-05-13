from rl_autoschedular import config as cfg
from rl_autoschedular.state import (
    OperationState, BenchmarkFeatures, OperationFeatures,
    OperationType, NestedLoopFeatures
)
from typing import Optional, Union, Literal
from rl_autoschedular.observation import extract_bench_features_from_file, build_op_features_vector
from rl_autoschedular.transforms import apply_transformation, is_vectorizable
from rl_autoschedular.evaluation import evaluate_code
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
    tmp_file: str
    """The temporary file to store the intermediate representations."""

    __bench_index: int
    """The index of the current benchmark."""

    def __init__(self, tmp_file: Optional[str] = None):
        """Initialize the environment.

        Args:
            tmp_file (Optional[str]): The temporary file to store the intermediate representations. Defaults to None.
        """
        # Generate a random file to be used in order to apply the transformations and evaluate the code
        # This is done in order to enable having multiple experiments at the same time, by letting each
        # experiment use a separate unique file to read and write intermediate representations
        if tmp_file is None:
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            tmp_file = f"tmp-debug/{random_str}.mlir" if cfg.debug else f"tmp/{random_str}.mlir"
        with open(tmp_file, "w") as file:
            file.write("")
        self.tmp_file = tmp_file

        # Load benchmark names and execution times from json file
        with open(cfg.json_file, "r") as file:
            benchmarks_json: dict[str, int] = json.load(file)
        # Build benchmark features
        self.benchmarks_data = []
        for bench_name, root_exec_time in tqdm(benchmarks_json.items(), desc="Extracting benchmark features", unit="bench"):
            bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
            benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)
            self.benchmarks_data.append(benchmark_data)

    def reset(self, bench_idx: Optional[int] = None, operation_idx: Optional[int] = None) -> tuple[OperationState, torch.Tensor]:
        """Reset the environment.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            OperationState: The initial state of the environment.
            torch.Tensor: The observation vector of the initial state.
        """
        # Get the benchmark
        if bench_idx is not None:
            self.__bench_index = bench_idx
        else:
            self.__bench_index = random.randint(0, len(self.benchmarks_data) - 1)

        return self.__init_op_state(-1 if operation_idx is None else operation_idx)

    def step(self, state: OperationState, raw_action: tuple[str, Optional[Union[list[int], int]]]) -> tuple[OperationState, torch.Tensor, float, bool, Optional[float]]:
        """Take a step in the environment.

        Args:
            state (OperationState): The current state.
            raw_action (tuple[str, Optional[Union[list[int], int]]]): The raw action to take.

        Returns:
            OperationState: The new state.
            Tensor: The observation vector of the new state.
            float: The reward of the action.
            bool: A flag indicating if the operation is done.
            Optional[float]: The speedup (if the operation is executed successfully) for logging purposes.
        """
        # Copy the current state to introduce the changes throughout the function
        next_state = state.copy()

        # Process the raw action
        transformation, parameters = self.__process_action(raw_action, next_state)

        # Attempt to apply the transformation to the code
        # - If the transformation fails: punish the agent, reset the code, and mark the operation as done
        new_transformed_code, trans_succeeded = self.__handle_transformation(transformation, parameters, state)
        if not trans_succeeded:
            print_error("Transformation Failed:", transformation, parameters)
            reward = self.__action_reward(trans_succeeded)
            self.__remove_invalid_trans(next_state)
            return next_state, self.__get_obs(next_state), reward, True, 1.0

        # Register the new code (transformation succeeded)
        next_state.transformed_code = new_transformed_code

        # Update the state infos to reflect the transformation
        self.__update_state_infos(next_state, transformation, parameters)

        # The operation is done if:
        # - The transformation is no_transformation or vectorization
        # - Maximum number of steps is reached
        op_done = transformation in ['no_transformation', 'vectorization'] or next_state.step_count == cfg.truncate

        # If the operation is not done, return the updated state with a reward of 0
        if not op_done:
            return next_state, self.__get_obs(next_state), 0.0, False, None

        # Evaluate the code (since the operation is done)
        try:
            new_exec_time, exec_succeeded = evaluate_code(next_state, self.__current_bench_data)
            if isinstance(exec_succeeded, Exception):
                raise exec_succeeded
            if not exec_succeeded or new_exec_time is None:
                raise Exception("Execution failed")
        except Exception as e:
            print_error(f"Error while evaluating the code: {e}")
            print_info("Bench:", next_state.bench_name)
            print_info("Transformations:", next_state.transformation_history)
            exec_succeeded = False
            new_exec_time = None

        # Next state and reward will take into consideration whether execution succeeded or not
        # i.e: if execution failed: punish the agent, reset the code, and mark the operation as done
        if cfg.sparse_reward:
            # Sparse reward: reward is given only if the benchmark is done
            # and it's calculated compared to the root execution time
            if self.__bench_is_done(next_state):
                reward = self.__action_reward(trans_succeeded, exec_succeeded, new_exec_time, self.__current_bench_data.root_exec_time)
            else:
                reward = 0.0
        else:
            reward = self.__action_reward(trans_succeeded, exec_succeeded, new_exec_time, next_state.exec_time)
        speedup = (self.__current_bench_data.root_exec_time / new_exec_time) if new_exec_time is not None else 1.0

        # Update the state infos to reflect the execution
        self.__update_state_exec_infos(next_state, new_exec_time)

        return next_state, self.__get_obs(next_state), reward, True, speedup

    def get_next_op_state(self, state: OperationState) -> tuple[Optional[OperationState], Optional[torch.Tensor], bool]:
        """Get the state that represents the next operation (can be from another benchmark).

        Args:
            state (OperationState): The current state.

        Returns:
            OperationState: The next state.
            torch.Tensor: The observation vector of the next state.
            bool: Flag indicating if the benchmark is done.
        """
        # Reset to another benchmark if the current benchmark is done (reached first operation)
        if self.__bench_is_done(state):
            return None, None, True

        # Build a new state that points to the next operation
        new_op_index = self.__current_op_index(state) - 1
        new_op_tag = self.__current_bench_data.operation_tags[new_op_index]
        new_op_features = self.__current_bench_data.operations[new_op_tag]
        new_action_mask = self.__init_action_mask(new_op_features)
        new_actions = self.__init_action_history()
        next_state = OperationState(
            bench_name=state.bench_name,
            operation_tag=new_op_tag,  # New operation tag
            operation_features=new_op_features,  # New operation features
            validated_code=state.validated_code,
            transformed_code=state.transformed_code,
            actions=new_actions,  # Empty actions history
            action_mask=new_action_mask,  # New action mask
            step_count=0,  # Reset step count
            exec_time=state.exec_time,
            transformation_history=state.transformation_history,
            interchange_permutation=[],  # Reset interchange permutation
            tmp_file=self.tmp_file,
        )

        return next_state, self.__get_obs(next_state), False

    def __init_op_state(self, operation_idx: int) -> tuple[OperationState, torch.Tensor]:
        """Create a new operation state.

        Args:
            operation_idx (int): The operation index.

        Returns:
            OperationState: The new operation state.
            torch.Tensor: The observation vector of the new operation state.
        """
        operation_tag = self.__current_bench_data.operation_tags[operation_idx]
        operation_features = self.__current_bench_data.operations[operation_tag]

        # Build action mask
        action_mask = self.__init_action_mask(operation_features)

        # Create empty action history
        actions = self.__init_action_history()

        state = OperationState(
            bench_name=self.__current_bench_data.bench_name,
            operation_tag=operation_tag,
            operation_features=operation_features.copy(),
            validated_code=self.__current_bench_data.code,
            transformed_code=self.__current_bench_data.code,
            actions=actions,
            action_mask=action_mask,
            step_count=0,
            exec_time=self.__current_bench_data.root_exec_time,
            transformation_history=[],
            interchange_permutation=[],
            tmp_file=self.tmp_file,
        )

        return state, self.__get_obs(state)

    @property
    def __current_bench_data(self) -> BenchmarkFeatures:
        """Get the current benchmark data.

        Returns:
            BenchmarkFeatures: The current benchmark data.
        """
        return self.benchmarks_data[self.__bench_index]

    def __current_op_index(self, state: OperationState) -> int:
        """Get the index of the current operation.

        Args:
            state (OperationState): The current state.

        Returns:
            int: The index of the current operation.
        """
        return self.__current_bench_data.operation_tags.index(state.operation_tag)

    def __bench_is_done(self, state: OperationState) -> bool:
        """Check if the benchmark is done.

        Args:
            state (OperationState): The current state.

        Returns:
            bool: A flag indicating if the benchmark is done.
        """
        return self.__current_op_index(state) == 0

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
        """Get the vector representing next interchange level

        Args:
            state (OperationState): The current state.

        Returns:
            np.ndarray: The vector representation of the interchange permutation.
        """
        next_loop = len(state.interchange_permutation)
        assert next_loop < len(state.operation_features.nested_loops)
        # perm_vector = np.zeros(cfg.max_num_loops, dtype=np.bool_)
        # perm_vector[next_loop] = 1
        # return perm_vector
        return (np.arange(cfg.max_num_loops) == next_loop).astype(np.bool_)

    def __init_action_mask(self, operation_features: OperationFeatures) -> np.ndarray:
        """Initialize the action mask.

        Notes:
            Action mask (NUM_TRANSFORMATIONS + L * (TS + 1) + L * (TS + 1) + interchange_mask):
                Transformations: no_transform, TP, T, I, vect
                TP: L loops * (TS + 1)
                T : L loops * (TS + 1)
                interchange_mask: 3 * L - 6 | L | 0

        Args:
            operation_features (OperationFeatures): The operation features.

        Returns:
            np.ndarray: The initialized action mask.
        """
        num_loops = len(operation_features.nested_loops)

        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        TP_BEGIN = cfg.num_transformations
        T_BEGIN = TP_BEGIN + L * (TS + 1)
        I_BEGIN_1C = T_BEGIN + L * (TS + 1)
        I_BEGIN_2C = I_BEGIN_1C + L - 1
        I_BEGIN_3C = I_BEGIN_2C + L - 2

        match cfg.interchange_mode:
            case 'enumerate':
                interchange_mask = 3 * L - 6
            case 'pointers':
                interchange_mask = L
            case 'continuous':
                interchange_mask = 0

        action_mask = np.ones((cfg.num_transformations + 2 * L * (TS + 1) + interchange_mask), dtype=bool)
        action_mask[:cfg.num_transformations] = cfg.init_action_mask
        action_mask[TP_BEGIN:T_BEGIN] = self.__tiling_mask(operation_features.nested_loops, for_parallelization=True)
        action_mask[T_BEGIN:I_BEGIN_1C] = self.__tiling_mask(operation_features.nested_loops, for_parallelization=False)

        if cfg.interchange_mode == 'enumerate':
            action_mask[I_BEGIN_1C + max(num_loops - 1, 0):I_BEGIN_2C] = False
            action_mask[I_BEGIN_2C + max(num_loops - 2, 0):I_BEGIN_3C] = False
            action_mask[I_BEGIN_3C + max(num_loops - 3, 0):] = False
        elif cfg.interchange_mode == 'pointers':
            action_mask[I_BEGIN_1C + num_loops:] = False

        # If we have only one loop -> Allow the first candidate which will be the identity permutation
        if num_loops == 1 and cfg.interchange_mode == 'enumerate':
            action_mask[I_BEGIN_1C] = True

        action_mask = self.__ensure_feasible_vectorization(action_mask, operation_features)

        return action_mask

    def __update_action_mask(self, action_mask: np.ndarray, transformation: str, parameters: list[int], operation_features: OperationFeatures) -> np.ndarray:
        """Update the action mask based on the transformation applied.

        Args:
            action_mask (np.ndarray): The current action mask.
            transformation (str): The transformation applied.
            parameters (list[int]): The parameters of the transformation.
            nested_loops (list[NestedLoopFeatures]): The nested loops features.

        Returns:
            np.ndarray: The updated action mask.
        """
        num_loops = len(operation_features.nested_loops)
        new_action_mask = action_mask.copy()

        N = cfg.num_transformations
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        TP_BEGIN = N
        T_BEGIN = TP_BEGIN + L * (TS + 1)
        I_BEGIN = T_BEGIN + L * (TS + 1)

        if cfg.interchange_mode == 'pointers':
            # Reset pointer masking
            new_action_mask[I_BEGIN:I_BEGIN + num_loops] = True

        match transformation:
            case 'parallelization':
                new_action_mask[:N] = [not cfg.force_vector, False, False, False, True]
            case 'tiling':
                new_action_mask[:N] = [False, False, False, True, False]
            case 'interchange':
                if 0 < len(parameters) < num_loops:
                    # In case of incomplete interchange, prevent any other action, and prevent repeating a loop
                    new_action_mask[:N] = [False, False, False, True, False]
                    for param in parameters:
                        new_action_mask[I_BEGIN + param] = False
                else:
                    new_action_mask[:N] = [False, True, False, False, False]

        # Update tiling masks
        new_action_mask[TP_BEGIN:T_BEGIN] = self.__tiling_mask(operation_features.nested_loops, for_parallelization=True)
        new_action_mask[T_BEGIN:I_BEGIN] = self.__tiling_mask(operation_features.nested_loops, for_parallelization=False)

        # If we have only one loop -> Allow the first candidate which will be the identity permutation
        if num_loops == 1 and cfg.interchange_mode == 'enumerate':
            new_action_mask[I_BEGIN] = True

        new_action_mask = self.__ensure_feasible_vectorization(new_action_mask, operation_features)

        return new_action_mask

    def __tiling_mask(self, nested_loops: list[NestedLoopFeatures], for_parallelization: bool) -> np.ndarray:
        """Get the tiling mask for the operation.

        Args:
            nested_loops (OperationFeatures): The operation features.
            for_parallelization (bool): A flag indicating if the tiling is for parallelization.

        Returns:
            np.ndarray: The tiling mask.
        """
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes

        num_loops = len(nested_loops)

        tile_sizes_counts = [
            self.__get_tiling_candidates(loop.upper_bound, loop.iterator_type, for_parallelization)[1] + 1
            for loop in nested_loops
        ]

        mask = np.ones((L * (TS + 1)), dtype=bool)
        mask[num_loops * (TS + 1):] = False
        mask = mask.reshape((L, TS + 1))

        for i, ts_count in enumerate(tile_sizes_counts):
            mask[i, ts_count:] = False

        return mask.reshape(-1)

    def __ensure_feasible_vectorization(self, action_mask: np.ndarray, operation_features: OperationFeatures) -> np.ndarray:
        """Automatically disable unfeasible vectorization if we should to.

        Args:
            action_mask (np.ndarray): The action mask.
            operation_features (OperationFeatures): The operation features.

        Returns:
            np.ndarray: The updated action mask.
        """
        new_action_mask = action_mask.copy()

        # Nothing to do, If Vectorization is already disabled
        if not new_action_mask[4]:
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
        if len(parameters) < num_loops or transformation in ['no_transformation', 'vectorization']:
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
        op_type_vector = self.__get_op_type_vector(state.operation_features.operation_type)
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
            # Get loop upper bounds
            candidates = [
                [0] + self.__get_tiling_candidates(loop.upper_bound, loop.iterator_type, action_name == 'parallelization')[0]
                for loop in state.operation_features.nested_loops
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

    def __get_tiling_candidates(self, n: int, iterator: Literal['parallel', 'reduction'], for_parallelization: bool) -> tuple[list[int], int]:
        """Get the tiling candidates for a given loop.

        Args:
            n (int): The loop upper bound.
            iterator_type (Literal['parallel', 'reduction']): The type of the loop iterator.
            for_parallelization (bool): A flag indicating if the tiling is for parallelization.

        Returns:
            list[int]: The tiling candidates.
            int: The number of candidates.
        """
        num_candidates = cfg.num_tile_sizes

        # No tiling is possible if:
        # - There is only one or no iteration
        # - The iterator is reduction and the tiling is for parallelization
        if n <= 1 or (iterator == 'reduction' and for_parallelization):
            return [0] * num_candidates, 0

        # Get the factors of the loop upper bound
        factors = []
        f = 1
        f_counts = 0
        while len(factors) < num_candidates:
            if f >= n:
                factors += [0] * (num_candidates - len(factors))
                break

            if n % f == 0:
                factors.append(f)
                f_counts += 1
            f *= 2

        return factors, f_counts

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
            print_error(f"Invalid interchange parameter: {x}")
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
            case 'parallelization' | 'tiling':
                # Apply the transformation and get the new code
                try:
                    transformed_code = apply_transformation(
                        state=state,
                        code=state.transformed_code,
                        transformation=transformation,
                        parameters=parameters,
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
                        )
                    except Exception as e:
                        print_error(f"Error while applying the transformation: {e}")
                        return '', False
                else:
                    # We keep the same code as previously if the permutation is not complete
                    transformed_code = state.transformed_code

            case 'no_transformation' | 'vectorization':
                if state.operation_features.operation_type == 'pooling':
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

        # Get updated operation features
        state.operation_features = self.__update_operation_features(state.operation_features, transformation, parameters)

        # Update the action mask
        state.action_mask = self.__update_action_mask(state.action_mask, transformation, parameters, state.operation_features)

        # If interchange permutation is not complete
        #  -> Record it in the state, and return
        if transformation == 'interchange' and len(parameters) < num_loops:
            state.interchange_permutation = parameters
            return

        # Erase saved interchange permutation (Not needed anymore)
        state.interchange_permutation = []

        # Register the action in the history
        state.actions = self.__update_action_history(state, transformation, parameters)
        state.transformation_history.append((transformation, parameters))

        # Update the step count
        state.step_count += 1

    def __update_operation_features(self, operation_features: OperationFeatures, transformation: str, parameters: list[int]) -> OperationFeatures:
        """Update the operation features after applying a transformation.

        Args:
            operation_features (OperationFeatures): The operation features.
            transformation (str): The transformation name.
            parameters (list[int]): The transformation parameters.

        Returns:
            OperationFeatures: The updated operation features.
        """
        new_operation_features = operation_features.copy()
        num_loops = len(new_operation_features.nested_loops)

        if transformation in ['no_transformation', 'vectorization']:
            return new_operation_features

        match transformation:
            case 'parallelization' | 'tiling':
                for nested_loop, tile_size in zip(new_operation_features.nested_loops, parameters):
                    if tile_size == 0:
                        continue
                    nested_loop.upper_bound = tile_size
            case 'interchange':
                if len(parameters) == num_loops:
                    for i, j in enumerate(parameters):
                        new_operation_features.nested_loops[i] = operation_features.nested_loops[j]
            case _:
                raise ValueError(f"Invalid transformation: {transformation}")

        return new_operation_features

    def __update_state_exec_infos(self, state: OperationState, new_exec_time: Optional[int]):
        """Update the state execution infos after evaluating the code.

        Args:
            state (OperationState): The current state.
            new_exec_time (Optional[int]): The new execution time.
        """
        # If the execution failed, reset the transformation sequence
        if new_exec_time is None:
            self.__remove_invalid_trans(state)
            return

        operation_idx = self.__current_op_index(state)

        # Mark the code as validated
        state.validated_code = state.transformed_code

        # Update the execution time
        state.exec_time = new_exec_time

        # Seal the transformation sequence
        state.transformation_history += [('done', [operation_idx])]

    def __remove_invalid_trans(self, state: OperationState):
        """Remove the latest invalid transformations and reset the transformation sequence.

        Args:
            state (OperationState): The current state.
        """
        # Get the last validated history
        last_op_idx = state.last_op_history_index()
        if last_op_idx is None:
            # It means there are no loose transformations (without 'done' at the end)
            validated_history = state.transformation_history
        else:
            # Get all transformations up until the last 'done'
            validated_history = state.transformation_history[:last_op_idx]

        # Reset the code to the last validated code
        state.transformed_code = state.validated_code

        # Reset the transformation sequence
        state.transformation_history = validated_history
