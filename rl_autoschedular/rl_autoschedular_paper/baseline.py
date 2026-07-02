"""Baseline execution time measurement for MLIR benchmarks.

This script measures the execution time of MLIR code without any transformations.
It executes the evaluation codes and updates the execution times (in nanoseconds)
in the current eval json file.

Execution Command:
    ```bash
    CONFIG_FILE_PATH=path/to/config.json poetry run baseline
    ```
"""

from dotenv import load_dotenv

load_dotenv(override=True)
load_dotenv('.env.debug')

from rl_autoschedular_paper.execution import Execution
from rl_autoschedular_paper.utils.config import Config
from mlir._mlir_libs._mlir.ir import Module, Context  # type: ignore
from tqdm import tqdm
import json
import os


def main():
    cfg = Config()
    exec = Execution("")

    if not cfg.eval_json_file:
        raise ValueError("Evaluation json file is not provided in the config file")
    with open(cfg.eval_json_file) as file:
        benchmarks_json: dict[str, int] = json.load(file)
    data_tqdm = tqdm(benchmarks_json, unit='file')
    for bench in data_tqdm:
        data_tqdm.set_postfix_str(bench)
        full_path = os.path.join(cfg.benchmarks_folder_path, bench + ".mlir")
        with open(full_path, 'r') as f:
            code = f.read()
        with Context():
            module = Module.parse(code)
        try:
            et, _, _, _ = exec.execute_code(module, bench, [])
        except Exception as e:
            print(f"Failed to execute {bench}: {e}")
            et = -1
        benchmarks_json[bench] = et

        try:
            with open(cfg.eval_json_file + '.tmp', 'w') as f:
                json.dump(benchmarks_json, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(cfg.eval_json_file + '.tmp', cfg.eval_json_file)
        except Exception as e:
            print(f"Failed to update {cfg.eval_json_file}: {e}")
        finally:
            if os.path.exists(cfg.eval_json_file + '.tmp'):
                os.remove(cfg.eval_json_file + '.tmp')

    print(f"Execution times updated in {cfg.eval_json_file}")

if __name__ == "__main__":
    main()
