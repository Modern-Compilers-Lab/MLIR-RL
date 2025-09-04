from rl_autoschedular.state import OperationState, BenchmarkFeatures
from rl_autoschedular.benchmarks import Benchmarks
from typing import Optional
from rl_autoschedular.execution import Execution
from rl_autoschedular.actions import Action, TiledFusion
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

        return self.__init_op_state(-1)

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
        action_failed = False
        try:
            self.__update_state_infos(next_state, action)
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
            action_failed = True

        # Check if state is terminal
        next_state.terminal = action.terminal or action_failed or next_state.step_count == Config().truncate

        return next_state

    def get_next_op_state(self, state: OperationState) -> Optional[OperationState]:
        """Get the state that represents the next operation (None if benchmark is done).

        Args:
            state (OperationState): The current state.

        Returns:
            Optional[OperationState]: The next state. If None then bench is done.
        """
        # Reset to another benchmark if the current benchmark is done (reached first operation)
        if self.__bench_is_done(state):
            return None

        # Build a new state that points to the next operation
        next_state = self.__init_op_state(self.__current_op_index(state) - 1)

        # Keep track of the transformation history
        next_state.transformation_history += state.transformation_history

        return next_state

    def apply_and_run_sequence(self, seq: list[list[Action]]) -> tuple[list[float], float, Optional[int], bool]:
        transformed_code, rewards = self.__apply_sequence(seq)

        # Evaluate the code (since the operation is done)
        try:
            new_exec_time, exec_succeeded, cache_miss = Execution().execute_code(transformed_code, self.benchmark_data.bench_name, seq)
            if not exec_succeeded:
                raise Exception("Incorrect results")
        except Exception as e:
            seq_str = '\n'.join([str(list(map(str, op_seq))) for op_seq in seq])
            print_error(
                "Error while evaluating the code\n"
                f"Error: {e}\n"
                f"Exception type: {type(e).__name__}\n"
                f"Call stack: {traceback.format_exc()}\n"
                f"Benchmark: {self.benchmark_data.bench_name}\n"
                f"Transformations:\n{seq_str}"
            )
            new_exec_time = None
            exec_succeeded = False
            cache_miss = True

        # The reward will take into consideration whether execution succeeded or not
        rewards[-1] = self.__action_reward(True, exec_succeeded, new_exec_time, self.benchmark_data.root_exec_time)
        speedup = (self.benchmark_data.root_exec_time / new_exec_time) if new_exec_time is not None else 1.0

        return rewards, speedup, new_exec_time, cache_miss

    def failed_seq(self, seq: list[list[Action]]) -> tuple[list[float], float, Optional[int], bool]:
        rewards = [0.0 for op_seq in reversed(seq) for action in op_seq for _ in range(len(action.sub_actions) + 1)]
        rewards[-1] = self.__action_reward(True, False)
        return rewards, 1.0, None, True

    def __init_op_state(self, operation_idx: int) -> OperationState:
        """Create a new operation state.

        Args:
            operation_idx (int): The operation index.

        Returns:
            OperationState: The new operation state.
            torch.Tensor: The observation vector of the new operation state.
        """
        operation_tag = self.benchmark_data.operation_tags[operation_idx]
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
            transformation_history=[[]],
            terminal=False,
        )

        return state

    def __current_op_index(self, state: OperationState) -> int:
        """Get the index of the current operation.

        Args:
            state (OperationState): The current state.

        Returns:
            int: The index of the current operation.
        """
        return self.benchmark_data.operation_tags.index(state.operation_tag)

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

        # if old < new * 2:
        #     return math.log(old / (new * 2))
        # else:
        #     return old / (new * 2) - 1
        return math.log10(old / new)

    def __update_state_infos(self, state: OperationState, action: Action):
        """Update state infos after applying a transformation.

        Notes: Updated fields are:
            - transformation_history
            - producers features in case of fusion
            - operation_features (to reflect the transformation)

        Args:
            state (OperationState): The current state.
            action (Action): The action taken.

        Returns:
            OperationState: The updated state.
        """
        # Record action
        state.record_action(action)

        # In case of fusion we need to update the producer features as well
        if isinstance(action, TiledFusion):
            action.update_producer_features(state, self.benchmark_data)

        # Get updated operation features
        state.operation_features = action.update_features(state.operation_features)

    def __apply_sequence(self, seq: list[list[Action]]) -> tuple[str, list[float]]:
        """Apply the sequence of actions to the state's code.

        Args:
            code (str): code to apply the actions to.
            seq (list[Action]): the sequence of actions to apply.

        Returns:
            tuple[str, list[float]]: the resulting code and rewards received for each action in the sequence.
        """
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
                    rewards.extend([self.__action_reward(False)] * rewards_count)
                    op_seq_already_failed = True
                    continue

                # Update transformed code
                transformed_code = new_transformed_code

                rewards.extend([0.0] * rewards_count)

        return transformed_code, rewards
