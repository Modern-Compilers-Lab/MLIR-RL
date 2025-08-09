from rl_autoschedular import config as cfg
from rl_autoschedular.state import OperationState, BenchmarkFeatures, extract_bench_features_from_file
from typing import Optional
from rl_autoschedular.evaluation import evaluate_code
from rl_autoschedular.actions import Action
from utils.log import print_error
from tqdm import tqdm
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
    is_training: bool
    """Flag indicating if the environment is in training mode or evaluation mode."""

    __bench_index: int
    """The index of the current benchmark."""

    def __init__(self, is_training: bool = True, tmp_file: Optional[str] = None):
        """Initialize the environment.

        Args:
            tmp_file (Optional[str]): The temporary file to store the intermediate representations. Defaults to None.
        """
        self.is_training = is_training

        # Generate a random file to be used in order to apply the transformations and evaluate the code
        if tmp_file is None:
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            tmp_file = f"tmp-debug/{random_str}.mlir" if cfg.debug else f"tmp/{random_str}.mlir"
        with open(tmp_file, "w") as file:
            file.write("")
        os.makedirs(tmp_file.replace(".mlir", ""), exist_ok=True)
        self.tmp_file = tmp_file

        # Load benchmark names and execution times from json file
        bench_json_file = cfg.json_file

        # If we are in evaluation mode, use the evaluation json file if provided
        if cfg.eval_json_file and not is_training:
            bench_json_file = cfg.eval_json_file

        with open(bench_json_file) as file:
            benchmarks_json: dict[str, int] = json.load(file)

        # Build benchmark features
        self.benchmarks_data = []
        for bench_name, root_exec_time in tqdm(benchmarks_json.items(), desc="Extracting benchmark features", unit="bench"):
            bench_file = os.path.join(cfg.benchmarks_folder_path, bench_name + ".mlir")
            benchmark_data = extract_bench_features_from_file(bench_name, bench_file, root_exec_time)

            if cfg.split_ops and is_training and len(benchmark_data.operation_tags) > 1:
                # Split benchmarks with more than one operation into multiple benchmarks
                for tag in benchmark_data.operation_tags:
                    # Create a new benchmark data with only the current operation
                    new_bench_data = benchmark_data.copy()
                    new_bench_data.bench_name = f"{benchmark_data.bench_name}_{tag}"
                    new_bench_data.operation_tags = [tag]
                    new_bench_data.operations = {tag: new_bench_data.operations[tag]}
                    self.benchmarks_data.append(new_bench_data)
            else:
                self.benchmarks_data.append(benchmark_data)

    def reset(self, bench_idx: Optional[int] = None) -> OperationState:
        """Reset the environment.

        Args:
            bench_idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            OperationState: The initial state of the environment.
            torch.Tensor: The observation vector of the initial state.
        """
        # Get the benchmark
        if bench_idx is not None:
            self.__bench_index = bench_idx
        else:
            self.__bench_index = random.randint(0, len(self.benchmarks_data) - 1)

        return self.__init_op_state(-1)

    def step(self, state: OperationState, action: Action) -> tuple[OperationState, float, bool, Optional[float]]:
        """Take a step in the environment.

        Args:
            state (OperationState): The current state.
            action (Action): The action to take.

        Returns:
            OperationState: The new state.
            float: The reward of the action.
            bool: A flag indicating if the operation is done.
            Optional[float]: The speedup (if the operation is executed successfully) for logging purposes.
        """
        # Copy the current state to introduce the changes throughout the function
        next_state = state.copy()

        # Attempt to apply the transformation to the code
        # - If the transformation fails: punish the agent, reset the code, and mark the operation as done
        new_transformed_code, trans_succeeded = action.apply(next_state)
        if not trans_succeeded:
            print_error("Transformation Failed:", action)
            reward = self.__action_reward(trans_succeeded)
            self.__remove_invalid_trans(next_state)
            return next_state, reward, True, 1.0

        # Register the new code (transformation succeeded)
        next_state.transformed_code = new_transformed_code

        # Update the state infos to reflect the transformation
        self.__update_state_infos(next_state, action)

        # The operation is done if:
        # - The transformation is terminal
        # - Maximum number of steps is reached
        op_done = action.terminal or next_state.step_count == cfg.truncate

        # If the operation is not done, return the updated state with a reward of 0
        if not op_done:
            return next_state, 0.0, False, None

        # Mark the state as terminal
        next_state.terminal = True

        # Evaluate the code (since the operation is done)
        try:
            new_exec_time, exec_succeeded = evaluate_code(next_state, self.__current_bench_data)
            if isinstance(exec_succeeded, Exception):
                raise exec_succeeded
            if not exec_succeeded or new_exec_time is None:
                raise Exception("Execution failed")
        except Exception as e:
            print_error(f"\n\nError while evaluating the code: {e}")
            print_error("Exception type:", type(e).__name__)
            print_error("Call stack:", e.__traceback__)
            print_error("Bench:", next_state.bench_name)
            print_error("Transformations:", next_state.transformation_history)
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

        return next_state, reward, True, speedup

    def get_next_op_state(self, state: OperationState) -> tuple[Optional[OperationState], bool]:
        """Get the state that represents the next operation (can be from another benchmark).

        Args:
            state (OperationState): The current state.

        Returns:
            OperationState: The next state.
            bool: Flag indicating if the benchmark is done.
        """
        # Reset to another benchmark if the current benchmark is done (reached first operation)
        if self.__bench_is_done(state):
            return None, True

        # Build a new state that points to the next operation
        new_op_index = self.__current_op_index(state) - 1
        new_op_tag = self.__current_bench_data.operation_tags[new_op_index]
        new_op_features = self.__current_bench_data.operations[new_op_tag]
        next_state = OperationState(
            bench_name=state.bench_name,
            operation_tag=new_op_tag,  # New operation tag
            operation_features=new_op_features,  # New operation features
            validated_code=state.validated_code,
            transformed_code=state.transformed_code,
            step_count=0,  # Reset step count
            exec_time=state.exec_time,
            transformation_history=[[]] + state.transformation_history,  # Start new sequence
            tmp_file=self.tmp_file,
            terminal=False,
        )

        return next_state, False

    def __init_op_state(self, operation_idx: int) -> OperationState:
        """Create a new operation state.

        Args:
            operation_idx (int): The operation index.

        Returns:
            OperationState: The new operation state.
            torch.Tensor: The observation vector of the new operation state.
        """
        operation_tag = self.__current_bench_data.operation_tags[operation_idx]
        operation_features = self.__current_bench_data.operations[operation_tag]

        state = OperationState(
            bench_name=self.__current_bench_data.bench_name,
            operation_tag=operation_tag,
            operation_features=operation_features.copy(),
            validated_code=self.__current_bench_data.code,
            transformed_code=self.__current_bench_data.code,
            step_count=0,
            exec_time=self.__current_bench_data.root_exec_time,
            transformation_history=[[]],
            tmp_file=self.tmp_file,
            terminal=False,
        )

        return state

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

        if old < new * 2:
            return math.log(old / (new * 2))
        else:
            return old / (new * 2) - 1
        # return math.log10(old / new)

    def __update_state_infos(self, state: OperationState, action: Action):
        """Update state infos after applying a transformation.

        Notes: Updated fields are:
            - operation_features (to reflect the transformation)
            - transformation_history
            - step _count

        Args:
            state (OperationState): The current state.
            action (Action): The action taken.

        Returns:
            OperationState: The updated state.
        """
        # Get updated operation features
        state.operation_features = action.update_features(state.operation_features)

        # Record action
        if state.step_count < len(state.transformation_history[0]):
            # Case where the last action should be replaced
            state.transformation_history[0][state.step_count] = action
        else:
            state.transformation_history[0].append(action)

        # Increase count only if action was applied
        if action.ready:
            state.step_count += 1

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

        # Mark the code as validated
        state.validated_code = state.transformed_code

        # Update the execution time
        state.exec_time = new_exec_time

    def __remove_invalid_trans(self, state: OperationState):
        """Remove the latest invalid transformations and reset the transformation sequence.

        Args:
            state (OperationState): The current state.
        """
        # Reset the code to the last validated code
        state.transformed_code = state.validated_code

        state.transformation_history[0] = []
