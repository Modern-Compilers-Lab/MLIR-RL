import os
from dotenv import load_dotenv
import json
import pathlib
from rl_autoschedular.execution import Execution
from utils.config import Config

load_dotenv(override=True)

config = Config()
cache_file = "cache/execution.json"
exec = Execution(exec_data_file=cache_file)

train_operations = {}
# eval_operations = {}

for benchmark in os.listdir(config.benchmarks_folder_path):
    benchmark_name = benchmark.split('.')[0]
    mlir_code_path = f"data/matmul/online_data/{benchmark}"
    mlir_code = pathlib.Path(mlir_code_path).read_text()
    time_ns, success, cache_miss = exec.execute_code(mlir_code, benchmark_name, seq=[])
    
    train_operations[benchmark_name] = time_ns
    # eval_operations[benchmark_name] = time_ns

    print(f"Benchmark: {benchmark_name}")
    print(f"Execution time: {time_ns} ns")
    print(f"Success: {success}, Cache miss: {cache_miss}")
    print("-" * 40)

# --- helper function to append safely ---
def append_json(file_path, new_data):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}
    # update old with new
    data.update(new_data)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

# append instead of overwrite
append_json(config.json_file, train_operations)
# append_json(config.eval_json_file, eval_operations)
