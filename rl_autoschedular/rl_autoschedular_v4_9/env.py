from rl_autoschedular_v4_9.state import OperationState, BenchmarkFeatures, OperationFeatures, IteratorType
from rl_autoschedular_v4_9.benchmarks import Benchmarks
from typing import Optional
from rl_autoschedular_v4_9.execution import Execution
from rl_autoschedular_v4_9.actions import Action, TiledFusion
from utils.log import print_error
from utils.config import Config
import random
import math
import traceback
import statistics


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
            old_operation_features = state.operation_features.copy()
            self.__update_state_infos(next_state, action)
            action.extras['shaped_reward'] = self.__shaped_reward(old_operation_features, next_state.operation_features, action)
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
            action.extras['shaped_reward'] = 0.0

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
            # Pass root_exec_time to allow profiling-based timeout safeguard
            cfg = Config()
            num_runs = cfg.eval_runs if hasattr(cfg, 'eval_runs') else 1
            run_times = []
            for run_idx in range(num_runs):
                run_time, run_ok, run_miss, run_err = Execution().execute_code(
                    transformed_code, 
                    self.benchmark_data.bench_name, 
                    seq,
                    root_exec_time=self.benchmark_data.root_exec_time
                )
                if run_ok:
                    run_times.append(run_time)
            
            if run_times:
                aggr = cfg.eval_aggregation if hasattr(cfg, 'eval_aggregation') else 'min'
                if aggr == 'median':
                    new_exec_time = statistics.median(run_times)
                elif aggr == 'mean':
                    new_exec_time = sum(run_times) / len(run_times)
                else:
                    new_exec_time = min(run_times)
                exec_succeeded = True
                cache_miss = run_miss
                error_msg = None
            else:
                raise Exception(run_err or "All execution runs failed")
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
        final_reward = self.__action_reward(True, exec_succeeded, new_exec_time, self.benchmark_data.root_exec_time)
        
        if not exec_succeeded:
            # Success-Contingent Reward Negation:
            # If the final execution fails (or times out), negate ALL intermediate shaped rewards
            # earned during the transformation sequence. This ensures the agent is only
            # rewarded for optimizations that result in a runnable, correct binary.
            rewards = [0.0] * len(rewards)
        
        rewards[-1] += final_reward
        # V4.6+: Slowdown penalty — zero intermediate rewards if speedup < 1.0, keep terminal
        if new_exec_time is not None and new_exec_time > 0 and self.benchmark_data.root_exec_time > 0:
            if (self.benchmark_data.root_exec_time / new_exec_time) < 1.0:
                if len(rewards) > 1:
                    rewards[:-1] = [0.0] * (len(rewards) - 1)
        speedup = (self.benchmark_data.root_exec_time / new_exec_time) if new_exec_time is not None else 0.0

        return rewards, speedup, new_exec_time, cache_miss

    def failed_seq(self, seq: list[list[Action]]) -> tuple[list[float], float, Optional[int], bool]:
        rewards = [0.0 for op_seq in reversed(seq) for action in op_seq for _ in range(len(action.sub_actions) + 1)]
        rewards[-1] = self.__action_reward(True, False)
        return rewards, 0.0, None, True

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
        new = max(new, 1)
        old = max(old, 1)
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
                # Account for sub-actions (incomplete steps) which were independently recorded in the trajectory
                for sub_action in action.sub_actions:
                    if op_seq_already_failed:
                        rewards.append(0.0)
                    else:
                        rewards.append(float(sub_action.extras.get('shaped_reward', 0.0)))

                if op_seq_already_failed:
                    rewards.append(0.0)
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
                    rewards.append(self.__action_reward(False))
                    op_seq_already_failed = True
                    continue

                # Update transformed code
                transformed_code = new_transformed_code

                shaped_reward = float(action.extras.get('shaped_reward', 0.0))
                rewards.append(shaped_reward)

        return transformed_code, rewards

    def __shaped_reward(self, old_features: OperationFeatures, new_features: OperationFeatures, action: Action) -> float:
        """Compute intermediate shaped reward from static feature changes.

        DISABLED in V4.9: This method always returns 0.0.
        Shaped reward caused entropy collapse in V4.x training.
        """
        return 0.0

    def __static_efficiency_score(self, features: OperationFeatures) -> float:
        cfg = Config()
        ai = self.__estimate_arithmetic_intensity(features)
        ai_score = math.log10(1.0 + ai)

        loop_count = max(1, len(features.nested_loops))
        parallel_loops = sum(loop.iterator_type == IteratorType.Parallel for loop in features.nested_loops)
        parallel_ratio = parallel_loops / loop_count

        vectorizable_score = 1.0 if features.vectorizable else 0.0

        return (
            cfg.reward_shaping_weight_ai * ai_score
            + cfg.reward_shaping_weight_vectorizable * vectorizable_score
            + cfg.reward_shaping_weight_parallel * parallel_ratio
        )

    def __estimate_arithmetic_intensity(self, features: OperationFeatures) -> float:
        loop_extent = 1.0
        index_upper_bounds = {}
        for loop in features.nested_loops:
            ub = max(1, int(loop.upper_bound))
            loop_extent *= ub
            index_upper_bounds[loop.arg] = ub

        total_ops = sum(features.op_count.values())
        flops = max(1.0, loop_extent * max(1, total_ops))

        bytes_moved = self.__estimate_bytes_moved(features, index_upper_bounds)
        return flops / max(1.0, bytes_moved)

    def __estimate_bytes_moved(self, features: OperationFeatures, index_upper_bounds: dict[str, int]) -> float:
        accesses = features.load_data + features.store_data
        if not accesses:
            return 1.0

        total_elements = 0.0
        for access in accesses:
            access_elements = 1.0
            for dim_expr in access:
                dim_extent = 1.0
                for index, ub in index_upper_bounds.items():
                    if index in dim_expr:
                        dim_extent *= ub
                access_elements *= max(1.0, dim_extent)
            total_elements += max(1.0, access_elements)

        # Use 4 bytes as default scalar width for stable shaping.
        return max(1.0, total_elements * 4.0)
