# test_execution.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os
import json
from rl_autoschedular.execution import Execution
from rl_autoschedular.actions import Action

def main():
    bench_name = "matmul"
    mlir_file_path = "./tests/benchmarks/matmul.mlir"  # path to your mlir file
    tmp_exec_file = "./tests/tmp_exec_data.json"       # temporary cache file

    # Ensure cache file exists
    if not os.path.exists(tmp_exec_file):
        with open(tmp_exec_file, "w") as f:
            json.dump({}, f)

    # Read MLIR code
    with open(mlir_file_path, "r") as f:
        mlir_code = f.read()

    print("=== Initializing Execution Manager ===")
    exec_manager = Execution(tmp_exec_file)

    # Example transformation sequence (empty for testing)
    seq = [[]]  # No transformations applied

    print("\n=== Executing MLIR Code ===")
    exec_time, success, cache_miss = exec_manager.execute_code(
        mlir_code,
        bench_name,
        seq
    )

    print(f"Execution time: {exec_time} ns")
    print(f"Success: {success}")
    print(f"Cache miss: {cache_miss}")

    '''
    print("\n=== Running again to test cache ===")
    exec_time_cached, success_cached, cache_miss_cached = exec_manager.execute_code(
        mlir_code,
        bench_name,
        seq
    )

    print(f"Cached execution time: {exec_time_cached} ns")
    print(f"Success: {success_cached}")
    print(f"Cache miss: {cache_miss_cached} (should be False)")

    print("\n=== Updating cache with dummy data ===")
    dummy_data = {bench_name: {exec_manager.get_code_cache_key(seq): exec_time}}
    exec_manager.update_execution_cache(dummy_data)
    print("Cache updated successfully.")
    '''

if __name__ == "__main__":
    if not os.path.exists("./tests/benchmarks/matmul.mlir"):
        print("ERROR: MLIR benchmark file not found. Please place matmul.mlir in ./tests/benchmarks/")
    else:
        main()
