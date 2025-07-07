import multiprocessing.managers
from aas import config as cfg
from aas.observation.benchmark import BenchmarkFeatures, extract_bench_features_from_file
from aas.state import OperationState
from aas.agent import AlphaAutoScheduler
from aas.wrappers import AASNetworkEstimation
from aas.evaluation import evaluate_code_with_timeout, evaluate_benchmark_code_with_timeout
from typing import Optional, Literal
import random
import string
import os
import json
from tqdm import tqdm
from collections import deque
import math
from utils.log import print_info, print_alert, print_success, print_error, FileLogger
import torch
import multiprocessing
import time
from copy import deepcopy

# To avoid leaking file descriptors when using multiprocessing
# https://github.com/pytorch/pytorch/issues/973#issuecomment-604473515
torch.multiprocessing.set_sharing_strategy('file_system')
# Set the number of threads for torch
torch.set_num_threads(4)
# Set the random seed for reproducibility
seed = 0
torch.manual_seed(seed)


class AASOpEnv:
    """Environment for training an AlphaAutoScheduler agent to optimize code by finding the best transformation list operation-wise."""

    benchmarks_data: list[BenchmarkFeatures]
    """List that contains all benchmarks features."""
    tmp_file_path: str
    """The temporary file to store the intermediate representations."""
    is_training: bool
    """Flag indicating if the environment is in training mode or evaluation mode."""

    def __init__(self, is_training: bool = True, tmp_file_path: Optional[str] = None):
        """Initialize the environment.

        Args:
            tmp_file_path (Optional[str]): The temporary file to store the intermediate representations. Defaults to None.
        """
        self.is_training = is_training

        # Generate a random file to be used in order to apply the transformations and evaluate the code
        if tmp_file_path is None:
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            tmp_file_path = f"tmp-debug/{random_str}.mlir" if cfg.debug else f"tmp/{random_str}.mlir"
        with open(tmp_file_path, "w") as file:
            file.write("")

        self.tmp_file_path = tmp_file_path

        # Get execution database path or generate a new one
        if cfg.use_cache and not cfg.exec_db_path:
            random_str = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
            cfg.exec_db_path = os.path.join('tmp', f'{random_str}.json')
            with open(cfg.exec_db_path, "w") as file:
                file.write("{}")

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
                    new_bench_data = deepcopy(benchmark_data)
                    new_bench_data.bench_name = f"{benchmark_data.bench_name}_{tag}"
                    new_bench_data.operation_tags = [tag]
                    new_bench_data.operations = {tag: new_bench_data.operations[tag]}
                    self.benchmarks_data.append(new_bench_data)
            else:
                self.benchmarks_data.append(benchmark_data)

    def reset(self, idx: Optional[int] = None, mode: Literal['random', 'sequential'] = 'random'):
        """Reset the environment.

        Args:
            idx (Optional[int]): The index of the benchmark to set the environement to. If None, a random benchmark is selected. Defaults to None.
            mode (Literal['random', 'sequential']): The mode to select the next benchmark. Defaults to 'random'.

        Returns:
            OperationState: The initial state of the environment.
        """
        if idx is not None:
            # We get the benchmark with the right index
            self.bench_index = idx
        else:
            if mode == 'sequential':
                # Get the next benchmark
                self.bench_index = (self.bench_index + 1) % len(self.benchmarks_data)
            else:
                # Get a random benchmark
                self.bench_index = random.randint(0, len(self.benchmarks_data) - 1)

        # Get benchmark data
        benchmark_data = self.benchmarks_data[self.bench_index]

        # Get the last operation
        operation_tag = benchmark_data.operation_tags[-1]
        operation_features = benchmark_data.operations[operation_tag]

        # Build the initial state
        state = OperationState(
            bench_features=benchmark_data,
            operation_tag=operation_tag,
            operation_features=operation_features,
            step_count=0,
            transformation_history=[]
        )

        # Return that state
        return state

    def step(self, state: OperationState, mode: Literal['random', 'sequential'] = 'random', optimization_mode: Optional[Literal['all', 'last']] = None):
        """Take a step in the environment given an agent.

        Args:
            state (OperationState): The current state of the environment.
            mode (Literal['random', 'sequential']): The mode to select the next operation. Defaults to 'random'.
            optimization_mode (Optional[Literal['all', 'last']]): The optimization mode to use. If None, defaults to cfg.optimization_mode.

        Returns:
            OperationState: The next state of the environment.
            bool: Whether the benchmark optimization is over or not.
        """

        # Get benchmark data
        bench_data = self.benchmarks_data[self.bench_index]

        # Set optimization mode
        if optimization_mode is None:
            optimization_mode = cfg.optimization_mode

        # Indicates that the benchmark optimization is over or not
        terminated = True
        if optimization_mode == "all":
            op_index = bench_data.operation_tags.index(state.operation_tag)
            if op_index > 0:
                # Benchmark optimization is not over
                terminated = False
                # Build a new state that points to the next operation
                new_op_tag = bench_data.operation_tags[op_index - 1]
                new_op_features = bench_data.operations[new_op_tag]
                next_state = OperationState(
                    bench_features=bench_data,
                    operation_tag=new_op_tag,
                    operation_features=new_op_features,
                    step_count=0,
                    transformation_history=[]
                )

        # If the benchmark optimization is over, we reset the environment
        if terminated:
            next_state = self.reset(mode=mode)

        return next_state, terminated

    def get_reward(self, state: OperationState, exec_time: int):
        """Get the reward for the agent given the execution time of the optimized code.

        Args:
            state (OperationState): The current state of the environment.
            exec_time (int): The execution time of the optimized code.

        Returns:
            float: The reward for the agent.
        """
        root_exec_time = state.bench_features.root_exec_time
        speedup = root_exec_time / exec_time if exec_time > 0 and root_exec_time > 0 else 1.0
        return math.log2(speedup)

    def eval(self, agent: AlphaAutoScheduler, mode: Literal['greedy', 'stochastic'] = 'greedy'):
        """Evaluate the agent on a set of benchmarks.

        Args:
            agent (AlphaAutoScheduler): The agent to evaluate.

        Returns:
            list[float]: The speedups reached by the agent.
        """
        # Run specified number of full benchmark episodes
        speedups = []
        state = self.reset(idx=0)
        for i in range(len(self.benchmarks_data)):
            optimized_states: list[OperationState] = []
            terminated = False
            while not terminated:
                # Run the agent on the current state
                optimized_state = agent.eval(state, mode=mode)
                # Save the last state in trajectory
                optimized_states.append(optimized_state)
                # Take a step in the environment
                state, terminated = self.step(state, mode='sequential', optimization_mode='all')
            # Evaluate the transformed code
            exec_time, assertion, _ = evaluate_benchmark_code_with_timeout(optimized_states, self.tmp_file_path, use_cache=cfg.use_cache)
            # Calculate speedup
            bench_features = optimized_states[0].bench_features
            root_exec_time = bench_features.root_exec_time
            print_info(f"Evaluation ({i + 1}/{len(self.benchmarks_data)}): {bench_features.bench_name}")
            if exec_time is not None and assertion:
                speedup = root_exec_time / exec_time
                print_success(f"Schedule: {BenchmarkFeatures.any_schedule_to_str([s.transformation_history for s in optimized_states])} - {speedup}")
                speedups.append(speedup)
        # Return the speedups
        return speedups


class AASTrainer:
    """Trainer for an AlphaAutoScheduler agent."""

    train_env: AASOpEnv
    """The environment to train the agent on."""
    eval_env: AASOpEnv
    """The environment to evaluate the agent on."""
    agent: AlphaAutoScheduler
    """The agent agent to train."""
    prev_agent: Optional[AlphaAutoScheduler]
    """The previous agent to compare with the current one."""
    data: deque[tuple[OperationState, AASNetworkEstimation]]
    """The data queue to store training data."""
    save_file_path: str
    """The path to save the agent to."""
    train_output_list: multiprocessing.managers.ListProxy
    """The list to store the output of the parallel MCTS searches."""
    file_logger: FileLogger
    """The file logger to log the training and evaluation data in files."""

    def __init__(self, env_type: Literal["op"] = "op", save_file_path: Optional[str] = None):
        """Initialize the trainer.

        Args:
            env (AASOpEnv): The environment to train the agent on.
            save_file_path (Optional[str]): The path to save the agent to. Defaults to None.
        """
        # Initialize the environment
        if env_type == "op":
            self.train_env = AASOpEnv()
            self.eval_env = AASOpEnv(is_training=False)
        else:
            raise ValueError(f"Invalid environment type ({env_type}). Please choose 'op' for operation-wise optimization.")
        # Initialize agents
        if save_file_path is None:
            self.save_file_path = os.path.join("models", "aas_agent.pt")
        else:
            self.save_file_path = save_file_path
        self.agent = AlphaAutoScheduler(self.train_env.get_reward)
        # self.agent.save(self.save_file_path)
        self.prev_agent = self.agent.copy()
        # Initialize data queue
        self.data = deque(maxlen=cfg.data_queue_max_length)
        self.prev_data = self.data.copy()
        # Initialize pipes for parallel MCTS searches
        manager = multiprocessing.Manager()
        self.train_output_list = manager.list()
        # Initialize file logger
        self.file_logger = FileLogger()
        if cfg.logging:
            self.file_logger.upload_dict('config.json', cfg.to_dict(), indent=2)

    def train(self):
        """Train the agent to optimize benchmarks."""
        # Initialize the environment
        prev_eval_speedups: Optional[list] = None
        exec_db: Optional[dict] = None
        train_state = self.train_env.reset()
        # Loop trough iterations
        for i in tqdm(range(cfg.nb_iterations), desc="Main Loop"):
            # Set agent temperature
            self.agent.set_temperature(1.0 if i < 500 else 0.5 if i < 750 else 0.25)
            # ======================== Gather training data ========================
            # Read execution database
            if cfg.use_cache and cfg.exec_db_path:
                with open(cfg.exec_db_path, "r") as f:
                    exec_db = json.load(f)
            # Start MCTS searches
            nb_cpus = multiprocessing.cpu_count()
            print_info(f"Started MCTS search on {nb_cpus} CPUs ...")
            mcts_start_time = time.time()
            # Clear the output list
            self.train_output_list[:] = []
            # Set number of torch threads to 1 to avoid issues with multiprocessing
            original_num_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            processes: list[multiprocessing.Process] = []
            # Run MCTS searches in parallel
            for id in range(cfg.nb_train_eps):
                # Run the agent on the current state
                process = multiprocessing.Process(target=self.agent.run_parallel, args=(id, self.train_output_list, (train_state, exec_db.get(train_state.bench_features.bench_name) if exec_db else None)))
                process.start()
                processes.append(process)
                train_state, _ = self.train_env.step(train_state)
            # Wait for all processes to finish
            for process in processes:
                process.join()
                process.close()
            # Reset the number of torch threads
            torch.set_num_threads(original_num_threads)
            trajectories: list[list[tuple[OperationState, AASNetworkEstimation]]] = [trajectory for _, trajectory in sorted(list(self.train_output_list), key=lambda x: x[0])]
            mcts_end_time = time.time()
            mcts_duration = mcts_end_time - mcts_start_time
            if cfg.logging:
                self.file_logger.append('train/mcts_duration', mcts_duration)
            print_info("Number of trajectories:", len(trajectories))
            print_info("Total number of data points:", sum(len(trajectory) for trajectory in trajectories))
            print_info(f"MCTS search ended in {mcts_duration} s ...")

            # ======================== Execute trajectories ========================
            print_info("Started execution ...")
            exec_start_time = time.time()
            speedups = []
            for j in tqdm(range(len(trajectories)), desc='Trajectories Execution'):
                # Get the last state of the trajectory
                last_state, _ = trajectories[j][-1]
                # Run the code with last state transformation list
                exec_time, assertion, transformed_code = evaluate_code_with_timeout(last_state, self.train_env.tmp_file_path, use_cache=cfg.use_cache)
                # Add the trajectory to the data queue
                if exec_time is not None and assertion:
                    # Get the reward
                    value = self.train_env.get_reward(last_state, exec_time)
                    speedups.append(last_state.bench_features.root_exec_time / exec_time)
                    for trajectory_state, trajectory_policy in trajectories[j]:
                        self.data.append((trajectory_state, AASNetworkEstimation(
                            policy=trajectory_policy,
                            value=value
                        )))
                else:
                    if exec_time is None:
                        if transformed_code:
                            print_error(f"EXECUTION ERROR: ({last_state.bench_features.bench_name} {last_state.operation_tag})")
                            print_error("ACTIONS:", last_state.transformation_history)
                        else:
                            print_error(f"TRANSFORMATION ERROR: ({last_state.bench_features.bench_name} {last_state.operation_tag})")
                            print_error("ACTIONS:", last_state.transformation_history)
                    else:
                        print_error(f"ASSERTION FAILED: ({last_state.bench_features.bench_name} {last_state.operation_tag})")
                        print_error("ACTIONS:", last_state.transformation_history)
            if cfg.logging:
                self.file_logger.extend('train/final_speedup', speedups)
            print_info("Average speedup:", sum(speedups) / len(speedups))
            print_info("Max speedup:", max(speedups))
            exec_end_time = time.time()
            exec_duration = exec_end_time - exec_start_time
            if cfg.logging:
                self.file_logger.append('train/execution_duration', exec_duration)
            print_info(f"Execution ended in {exec_duration} s ...")

            # ======================== Train the agent =============================
            # Train the current agent
            print_info("Started training ...")
            train_start_time = time.time()
            train_stats = self.agent.train(list(self.data))
            if cfg.logging:
                self.file_logger.extend('train/selection_loss', train_stats.selection_loss)
                self.file_logger.extend('train/value_loss', train_stats.value_loss)
                for j in range(cfg.max_num_loops):
                    self.file_logger.extend(f'train/parallel_params_loss_{j}', train_stats.parallel_params_loss[j])
            train_end_time = time.time()
            train_duration = train_end_time - train_start_time
            if cfg.logging:
                self.file_logger.append('train/train_duration', train_duration)
            print_info(f"Training ended in {train_duration} s ...")

            # ================ Evaluate the agent without MCTS =====================
            if i > 0 and i % 100 == 0:
                print_info("Started evaluation ...")
                eval_start_time = time.time()
                eval_speedups = self.eval_env.eval(self.agent, mode='greedy')
                avg_eval_speedup = sum(eval_speedups) / len(eval_speedups) if len(eval_speedups) > 0 else 0.0
                print_info("Greedy speedups average:", avg_eval_speedup)
                if cfg.logging:
                    self.file_logger.extend('eval/final_speedup', eval_speedups)
                    self.file_logger.append('eval/average_speedup', avg_eval_speedup)
                    bench_names = [bench.bench_name for bench in self.eval_env.benchmarks_data]
                    for bench_name, eval_speedup in zip(bench_names, eval_speedups):
                        self.file_logger.append(f'eval/speedup/{bench_name}', eval_speedup)
                if cfg.pitting:
                    score = self.pit(eval_speedups, prev_eval_speedups)
                    if score >= 0:
                        print_success("Agent improved over the previous one with a score of", score)
                        prev_eval_speedups = eval_speedups
                        self.prev_agent = self.agent.copy()
                        self.prev_data = self.data.copy()
                    else:
                        print_alert("Agent did not improve over the previous one with a score of", score)
                        self.agent = self.prev_agent
                        self.data = self.prev_data
                eval_end_time = time.time()
                eval_duration = eval_end_time - eval_start_time
                if cfg.logging:
                    self.file_logger.append('eval/eval_duration', eval_duration)
                print_info(f"Evaluation ended in {eval_duration} s ...")

            # ====================== Save the best agent ===========================
            # Save the best agent so far
            self.agent.save(self.save_file_path)

    def pit(self, speedups1: list[float], speedups2: list[float]):
        """Compare two agents on a set of benchmarks.

        Args:
            speedups1 (list[float]): The speedups of the first agent.
            speedups2 (list[float]): The speedups of the second agent.

        Returns:
            float: The score of the comparison.
        """
        # Initialize the score
        score = 0
        # Check if speedups are available
        if not speedups1 or not speedups2:
            return score
        # Calculate score based on ratio between speedups (positive score = keep new agent, negative score = keep old agent)
        for speedup1, speedup2 in zip(speedups1, speedups2):
            score += (speedup1 / speedup2 - 1.0) if speedup1 >= speedup2 else -(speedup2 / speedup1 - 1.0)
        # Return the score
        return score

    def get_best_agent(self) -> AlphaAutoScheduler:
        """Load the best agent so far.

        Returns:
            AlphaAutoScheduler: The best agent so far.
        """
        return AlphaAutoScheduler.load_from_file(self.save_file_path, self.train_env.get_reward)
