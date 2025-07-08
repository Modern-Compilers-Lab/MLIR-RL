from dotenv import load_dotenv
load_dotenv(override=True)
import os
os.environ['AAS_CONFIG_FILE_PATH'] = 'config/aas_all_no_cache.json'
from aas import config as cfg
from aas.env import AASOpEnv
from aas.agent import AlphaAutoScheduler
from aas.state import OperationState, BenchmarkFeatures
from aas.evaluation import evaluate_benchmark_code_with_timeout
from aas.action import ParameterizedAction, Vectorization, NoTransformation
from tqdm import tqdm

env = AASOpEnv(is_training=False)

agent: AlphaAutoScheduler = AlphaAutoScheduler.load_from_file('saves/aas_all_agent_no_cache.pt', env.get_reward)

results = {}
state = env.reset(idx=0)
for i in tqdm(range(len(env.benchmarks_data)), desc='Evaluation'):
    optimized_states: list[OperationState] = []
    terminated = False
    while not terminated:
        # Run the agent on the current state
        optimized_state = agent.eval(state, mode='greedy')
        # Save the last state in trajectory
        optimized_states.append(optimized_state)
        # Take a step in the environment
        state, terminated = env.step(state, mode='sequential', optimization_mode='all')
    # Get initial execution time and benchmark features
    bench_features = optimized_states[0].bench_features
    root_exec_time = bench_features.root_exec_time
    # Evaluate the transformed code
    exec_time, assertion, transformed_code = evaluate_benchmark_code_with_timeout(optimized_states, env.tmp_file_path, use_cache=cfg.use_cache)
    # Calculate speedup
    if exec_time is not None and assertion:
        speedup = root_exec_time / exec_time
        bench_name = bench_features.bench_name
        for s in optimized_states:
            for action in s.transformation_history:
                if isinstance(action, ParameterizedAction):
                    action.params = action.params[:len(s.operation_features.nested_loops)]
        transformation_history = BenchmarkFeatures.any_schedule_to_str([s.transformation_history for s in optimized_states])
        transformation_history = transformation_history.split('|')
        s = f"- Bench: {bench_name}\n"
        s += "- Schedule:\n"
        for j in range(len(transformation_history)):
            s += f"\t- operation_{len(transformation_history) - 1 - j}:\n"
            actions_str = transformation_history[j].split(')')
            if f"{Vectorization.DEFAULT_NAME}(" not in actions_str:
                actions_str.append(f"{NoTransformation.DEFAULT_NAME}(")
            for action in actions_str:
                if action:
                    s += f"\t\t- {action})\n"
        s += f"- Speedup: {speedup}"
        print(f'\033[92m{s}\033[0m')
        # Output code files
        f = open(f'demo/{bench_name}_original.mlir', 'w')
        f.write(bench_features.code)
        f.close()
        f = open(f'demo/{bench_name}_optimized.mlir', 'w')
        f.write(transformed_code)
        f.close()
        # Store the results
        results[bench_features.bench_name] = {
            'bench_features': bench_features,
            'speedup': speedup,
            'transformation_history': BenchmarkFeatures.any_schedule_to_str([s.transformation_history for s in optimized_states])
        }
