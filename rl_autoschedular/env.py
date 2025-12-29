import os
from rl_autoschedular.state import OperationState, BenchmarkFeatures, extract_bench_features_from_code
from rl_autoschedular.benchmarks import Benchmarks
from typing import Optional
from rl_autoschedular.execution import Execution
from rl_autoschedular.actions import Action, TiledFusion
from utils.file_logger import FileLogger
from utils.log import print_error
from utils.config import Config
import random
import math
import traceback


class Env:
    """RL Environment class"""

    bench_idx: int
    """Index of the selected benchmark"""
    benchmark_data: BenchmarkFeatures
    """Features of the selected benchmark"""
    processed_tags: set[str]
    """List of tags that have already been processed"""

    def __new__(cls):
        # If intermediate_transformations is enabled instantiate reactive environment, else predictive
        if Config().intermediate_transforms:
            return super().__new__(ReactiveEnv)
        else:
            return super().__new__(PredictiveEnv)

    def reset(self, benchs: Benchmarks, bench_idx: Optional[int] = None) -> OperationState:
        """Reset the environment.

        Args:
            bench_idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.

        Returns:
            OperationState: The initial state of the environment.
        """
        # Get the benchmark
        if bench_idx is None:
            bench_idx = random.randint(0, len(benchs) - 1)
        self.bench_idx = bench_idx
        self.benchmark_data = benchs[bench_idx].copy()
        self.processed_tags = set()

        return self._init_op_state(self.benchmark_data.operation_tags[-1])

    def step(self, state: OperationState, action: Action) -> OperationState:
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

        # Update the state infos to reflect the transformation
        try:
            self._update_state_infos(next_state, action)
        except Exception as e:
            seq_str = '\n'.join([str(list(map(str, op_seq))) for op_seq in state.transformation_history])
            print_error(
                'Error while expecting action effect\n'
                f"Action: {repr(action)}\n"
                f"Error: {e}\n"
                f"Call stack: {traceback.format_exc()}\n"
                f"Benchmark: {self.benchmark_data.bench_name}\n"
                f"Transformations:\n{seq_str}"
            )
            next_state.failed = True

        # Check if state is terminal
        next_state.terminal = action.terminal or next_state.failed or next_state.step_count == Config().truncate

        if next_state.terminal:
            assert next_state.operation_tag not in self.processed_tags
            self.processed_tags.add(next_state.operation_tag)

        return next_state

    def get_next_op_state(self, state: OperationState) -> Optional[OperationState]:
        """Get the state that represents the next operation (None if benchmark is done).

        Args:
            state (OperationState): The current state.

        Returns:
            Optional[OperationState]: The next state. If None then bench is done.
        """
        # Reset to another benchmark if the current benchmark is done (reached first operation)
        if self._bench_is_done:
            return None

        # Build a new state that points to the next operation
        next_state = self._init_op_state(self._unprocessed_tags[-1])

        # Keep track of the transformation history
        next_state.transformation_history += state.transformation_history

        return next_state

    def apply_and_run_sequence(self, seq: list[list[Action]]) -> tuple[list[float], float, Optional[int], bool]:
        transformed_code, rewards = self._apply_sequence(seq)

        # Evaluate the code (since the operation is done)
        try:
            new_exec_time, exec_succeeded, cache_miss = Execution().execute_code(transformed_code, self.benchmark_data.bench_name, seq)
            if not exec_succeeded:
                raise Exception("Incorrect results")
        except Exception as e:
            seq_str = '\n'.join([str(list(map(str, op_seq))) for op_seq in seq])
            print_error(
                "Error while executing the code\n"
                f"Error: {e}\n"
                f"Exception type: {type(e).__name__}\n"
                f"Call stack: {traceback.format_exc()}\n"
                f"Benchmark: {self.benchmark_data.bench_name}\n"
                f"Transformations:\n{seq_str}"
            )
            new_exec_time = None
            exec_succeeded = False
            cache_miss = True

            with open(os.path.join(FileLogger().run_dir, "errors.mlir"), "a") as f:
                seq_str = '\n// '.join([str(list(map(str, op_seq))) for op_seq in seq])
                f.write(f"// {self.benchmark_data.bench_name}\n")
                f.write("// EXEC ERROR\n")
                f.write(f"// {seq_str}\n")
                f.write(transformed_code)
                f.write("\n// -----\n")

        # The reward will take into consideration whether execution succeeded or not
        rewards[-1] = self._action_reward(True, exec_succeeded, new_exec_time, self.benchmark_data.root_exec_time)
        speedup = (self.benchmark_data.root_exec_time / new_exec_time) if new_exec_time is not None else 1.0

        return rewards, speedup, new_exec_time, cache_miss

    def failed_seq(self, seq: list[list[Action]]) -> tuple[list[float], float, Optional[int], bool]:
        rewards = [0.0 for op_seq in reversed(seq) for action in op_seq for _ in range(len(action.sub_actions) + 1)]
        rewards[-1] = self._action_reward(True, False)
        return rewards, 1.0, None, True

    @property
    def _unprocessed_tags(self) -> list[str]:
        """Get the list of unprocessed operations.

        Returns:
            The list of unprocessed operations.
        """
        return [op_tag for op_tag in self.benchmark_data.operation_tags if op_tag not in self.processed_tags]

    @property
    def _bench_is_done(self) -> bool:
        """Check if the benchmark is done.

        Args:
            state (OperationState): The current state.

        Returns:
            bool: A flag indicating if the benchmark is done.
        """
        return len(self._unprocessed_tags) == 0

    def _init_op_state(self, operation_tag: str) -> OperationState:
        """Create a new operation state.

        Args:
            operation_idx (int): The operation index.

        Returns:
            OperationState: The new operation state.
            torch.Tensor: The observation vector of the new operation state.
        """
        operation_features = self.benchmark_data.operations[operation_tag].copy()

        for action in operation_features.pre_actions:
            operation_features = action.update_features(operation_features)

        producer_tag = None
        producer_operand_idx = None
        producer_features = None
        if operation_features.producers:
            # NOTE: To change with mutliple producers support
            producer_tag = operation_features.producers[-1][0]
            # NOTE: To change with mutliple uses support
            producer_operand_idx = min(idx for t, idx in operation_features.producers if t == producer_tag)
            producer_features = self.benchmark_data.operations[producer_tag].copy()

        state = OperationState(
            bench_idx=self.bench_idx,
            bench_name=self.benchmark_data.bench_name,
            operation_tag=operation_tag,
            original_operation_features=self.benchmark_data.operations[operation_tag].copy(),
            operation_features=operation_features,
            producer_tag=producer_tag,
            producer_operand_idx=producer_operand_idx,
            producer_features=producer_features,
        )

        return state

    def _action_reward(self, trans_succeeded: bool, exec_succeeded: Optional[bool] = None, new_exec_time: Optional[int] = None, old_exec_time: Optional[int] = None) -> float:
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
        return self._speedup_reward(new_exec_time, old_exec_time)

    def _speedup_reward(self, new: int, old: int) -> float:
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

    def _update_state_infos(self, state: OperationState, action: Action):
        """Update state infos after applying a transformation.

        Args:
            state (OperationState): The current state.
            action (Action): The action taken.
        """
        # Record action
        state.record_action(action)

    def _apply_sequence(self, seq: list[list[Action]]) -> tuple[str, list[float]]:
        """Apply the sequence of actions to the state's code.

        Args:
            code (str): code to apply the actions to.
            seq (list[Action]): the sequence of actions to apply.

        Returns:
            tuple[str, list[float]]: the resulting code and rewards received for each action in the sequence.
        """
        raise NotImplementedError


class ReactiveEnv(Env):
    """Environment that applies transformations to the code in each step."""

    rewards_record: list[float]
    """Record of intermediate rewards"""

    def reset(self, benchs, bench_idx=None):
        self.rewards_record = []
        return super().reset(benchs, bench_idx)

    def step(self, state, action):
        next_state = super().step(state, action)

        if next_state.failed:
            seq_str = '\n// '.join([str(list(map(str, op_seq))) for op_seq in state.transformation_history])
            with open(os.path.join(FileLogger().run_dir, "errors.mlir"), "a") as f:
                f.write(f"// {self.benchmark_data.bench_name}\n")
                f.write(f"// Action: {repr(action)}\n")
                f.write(f"// {seq_str}\n")
                f.write(self.benchmark_data.code)
                f.write("\n// -----\n")

        # In a reactive env we know for sure if the action is
        # successful. However we don't know that in a predictive
        # env, that's why we use `state.failed` only here
        self.rewards_record.append(self._action_reward(False) if next_state.failed else 0.0)

        return next_state

    def _update_state_infos(self, state, action):
        super()._update_state_infos(state, action)

        new_transformed_code = action.apply(self.benchmark_data.code)
        self.benchmark_data = extract_bench_features_from_code(
            self.benchmark_data.bench_name,
            new_transformed_code,
            self.benchmark_data.root_exec_time,
            self.benchmark_data.tag_counter,
        )

        # Update state
        if action.terminal:
            return
        new_state = self._init_op_state(state.operation_tag)
        state.operation_features = new_state.operation_features
        state.producer_features = new_state.producer_features
        state.producer_tag = new_state.producer_tag
        state.producer_operand_idx = new_state.producer_operand_idx

    def _apply_sequence(self, seq):
        return self.benchmark_data.code, self.rewards_record


class PredictiveEnv(Env):
    """Environment that predicts the effect of transformations on the code."""

    def _update_state_infos(self, state, action):
        super()._update_state_infos(state, action)

        # In case of fusion we need to update the producer features as well
        if isinstance(action, TiledFusion):
            action.update_producer_features(state, self.benchmark_data)

        # Get updated operation features
        state.operation_features = action.update_features(state.operation_features)

    def _apply_sequence(self, seq):
        rewards: list[float] = []
        transformed_code = self.benchmark_data.code
        for op_seq in reversed(seq):
            op_seq_already_failed = False
            for action in op_seq:
                # We need to assign the same reward to all sub actions
                rewards_count = len(action.sub_actions) + 1

                if op_seq_already_failed:
                    rewards.extend([0.0] * rewards_count)
                    continue

                # Attempt to apply the transformation to the code
                # - If the transformation fails: punish the agent, reset the code, and mark the operation as done
                try:
                    new_transformed_code = action.apply(transformed_code)
                except Exception as e:
                    seq_str = '\n'.join([str(list(map(str, op_seq))) for op_seq in seq])
                    print_error(
                        f"Error applying action\n"
                        f"Action: {repr(action)}\n"
                        f"Error: {e}\n"
                        f"Benchmark: {self.benchmark_data.bench_name}\n"
                        f"Transformations:\n{seq_str}"
                    )
                    rewards.extend([0.0] * (rewards_count - 1) + [self._action_reward(False)])
                    op_seq_already_failed = True
                    continue

                # Update transformed code
                transformed_code = new_transformed_code

                rewards.extend([0.0] * rewards_count)

        return transformed_code, rewards
