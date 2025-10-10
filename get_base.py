from rl_autoschedular.execution import Execution
from tqdm import tqdm
import sys
import json
import os

if len(sys.argv) != 2:
    print("Usage: python get_base.py <path_to_folder>")
    sys.exit(1)
path_to_folder = sys.argv[1]
if not os.path.isdir(path_to_folder):
    print(f"Error: {path_to_folder} is not a valid directory.")
    sys.exit(1)

output_data = {}
exec = Execution("")

code_files = [f for f in os.listdir(path_to_folder) if f.endswith('.mlir')]
files_tqdm = tqdm(code_files, unit='file')
for code_file in files_tqdm:
    bench_name = code_file.replace('.mlir', '')
    files_tqdm.set_postfix_str(bench_name)
    full_path = os.path.join(path_to_folder, code_file)
    with open(full_path, 'r') as f:
        code = f.read()
    try:
        et, _, _ = exec.execute_code(code, bench_name, [])
    except Exception as e:
        print(f"Failed to execute {bench_name}: {e}")
        et = -1
    output_data[bench_name] = et

    with open('base_exec_times.json', 'w') as f:
        json.dump(output_data, f, indent=4)
