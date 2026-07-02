from tqdm import tqdm
import argparse
import traceback
import json
import os
import signal
from pathlib import Path

try:
    from mlir.ir import Context, Module, MemRefType
    from mlir.dialects.func import FuncOp
    MLIR_AVAILABLE = True
except ImportError:
    MLIR_AVAILABLE = False

from utils.implementation import (
    get_autoschedular_impl,
    get_base_file_path,
    import_autoschedular_module,
)

# Per-file timeout in seconds. Files that take longer are killed (et=-1).
FILE_TIMEOUT = 15


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("MLIR execution timed out")


parser = argparse.ArgumentParser(description="Get base execution times for MLIR benchmarks.")
parser.add_argument("--config", default=None, help="Path to config JSON (derives benchmarks-dir and output)")
parser.add_argument("--benchmarks-dir", default=None, help="Override: path to folder containing .mlir files")
parser.add_argument("--output", default=None, help="Override: path to output JSON file")
parser.add_argument("--implementation", default=None, help="Autoscheduler implementation package (default: AUTOSCHEDULER_IMPL or rl_autoschedular)")
parser.add_argument("--timeout", type=int, default=FILE_TIMEOUT, help=f"Per-file timeout in seconds (default: {FILE_TIMEOUT})")
parser.add_argument("--chunk-index", type=int, default=0, help="0-based index of the chunk to process")
parser.add_argument("--num-chunks", type=int, default=1, help="Total number of chunks to split the workload into")
args = parser.parse_args()

implementation = args.implementation or get_autoschedular_impl(config_path=args.config)
Execution = import_autoschedular_module("execution", implementation).Execution

if args.config:
    with open(args.config) as _f:
        _cfg = json.load(_f)
    path_to_folder = args.benchmarks_dir or _cfg["benchmarks_folder_path"]
    # If --output is explicitly given, use it; otherwise write to the
    # dataset-level generic base.json (shared across implementations).
    if args.output:
        output_path = args.output
    else:
        output_path = str(get_base_file_path(_cfg["results_dir"], implementation=None))
else:
    if not args.benchmarks_dir or not args.output:
        parser.error("Provide --config, or both --benchmarks-dir and --output")
    path_to_folder = args.benchmarks_dir
    output_path    = args.output

if not os.path.isdir(path_to_folder):
    print(f"Error: {path_to_folder} is not a valid directory.")
    raise SystemExit(1)

Path(output_path).parent.mkdir(parents=True, exist_ok=True)

if args.num_chunks > 1:
    old_out = Path(output_path)
    output_path = str(old_out.parent / f"{old_out.stem}_chunk{args.chunk_index}{old_out.suffix}")

if os.path.exists(output_path):
    try:
        with open(output_path, 'r') as f:
            output_data = json.load(f)
        print(f"Resuming from existing output file. Found {len(output_data)} completed benchmarks.")
    except Exception as e:
        print(f"Failed to load existing {output_path}: {e}")
        output_data = {}
else:
    output_data = {}

import os
os.environ["EXEC_TIMEOUT"] = str(args.timeout)

exec = Execution("")

print(f"Using autoscheduler implementation: {implementation}")

code_files = [f for f in os.listdir(path_to_folder) if f.endswith('.mlir')]
code_files = sorted(code_files)

# Partition the dataset
if args.num_chunks > 1:
    chunk_size = len(code_files) // args.num_chunks
    start_idx = args.chunk_index * chunk_size
    end_idx = start_idx + chunk_size if args.chunk_index < args.num_chunks - 1 else len(code_files)
    code_files = code_files[start_idx:end_idx]
    print(f"-- Processing chunk {args.chunk_index + 1}/{args.num_chunks} -- Files {start_idx} to {end_idx-1}")

# Only skip if the benchmark successfully ran (et > 0)
remaining_files = [f for f in code_files if output_data.get(f.replace('.mlir', ''), -1) <= 0]
print(f"Remaining benchmarks to process: {len(remaining_files)} / {len(code_files)}")

files_tqdm = tqdm(remaining_files, unit='file')
for code_file in files_tqdm:
    bench_name = code_file.replace('.mlir', '')
    files_tqdm.set_postfix_str(bench_name)
    full_path = os.path.join(path_to_folder, code_file)
    with open(full_path, 'r') as f:
        code = f.read()

    # Pre-check MLIR
    try:
        with Context():
            pre_module = Module.parse(code)
            pre_funcs = [op for op in pre_module.body.operations if isinstance(op, FuncOp)]
            pre_main = next((op for op in pre_funcs if op.name.value == 'main'), None)
            if pre_main is None:
                files_tqdm.write(f"Failed to execute {bench_name}: No main function found")
                output_data[bench_name] = -1
                with open(output_path, 'w') as f:
                    json.dump(output_data, f, indent=4)
                continue
    except Exception as e:
        files_tqdm.write(f"Failed to execute {bench_name}: Pre-check failed \u2014 {e}")
        output_data[bench_name] = -1
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        continue

    try:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(args.timeout)
        exec_results = exec.execute_code(code, bench_name, [])
        et = exec_results[0]
        signal.alarm(0)
    except TimeoutError:
        files_tqdm.write(f"Failed to execute {bench_name}: timed out after {args.timeout}s")
        et = -1
    except Exception as e:
        files_tqdm.write(f"Failed to execute {bench_name}: {e}")
        et = -1
    finally:
        signal.alarm(0)

    output_data[bench_name] = et
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
