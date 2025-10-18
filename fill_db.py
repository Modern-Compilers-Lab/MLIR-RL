from dotenv import load_dotenv

from utils.log import print_info, print_success
load_dotenv(override=True)
load_dotenv('.env.debug')

import json
import sys
from typing import Optional

from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.benchmarks import Benchmarks
from rl_autoschedular.env import Env
from rl_autoschedular.execution import Execution
from rl_autoschedular.observation import Observation

from utils.config import Config
from utils.dask_manager import DaskManager
from utils.file_logger import FileLogger


def execute_bench(bench_idx: int, exec_data_file: str, benchs: Benchmarks, main_exec_data: Optional[dict[str, dict[str, int]]]):
    exec = Execution(exec_data_file, main_exec_data)
    env = Env()
    state = env.reset(benchs, bench_idx)

    finalized = False
    while not finalized:
        assert not state.terminal
        obs = Observation.from_state(state)
        eps_distributions = ActionSpace.uniform_distributions(obs)
        action_index = ActionSpace.sample(obs, eps_distributions, eps_distributions, uniform=True)
        assert action_index.size(0) == 1
        action_index = action_index[0]
        action = ActionSpace.action_by_index(action_index, state)
        state = env.step(state, action)
        if state.terminal:
            next_op_state = env.get_next_op_state(state)
            if next_op_state is not None:
                state = next_op_state
            else:
                finalized = True

    cache_key = exec.get_code_cache_key(state.transformation_history)
    _, _, new_exec_time, cache_miss = env.apply_and_run_sequence(state.transformation_history)

    return new_exec_time, state.bench_name, cache_key, cache_miss


if __name__ == "__main__":
    dm = DaskManager()
    fl = FileLogger()
    cfg = Config()

    print_info(f"Config: {cfg}")
    print_success(f'Logging to: {fl.run_dir}')

    def load_eval_data():
        return Benchmarks()

    def load_main_exec_data() -> Optional[dict[str, dict[str, int]]]:
        main_exec_data = None
        if Config().main_exec_data_file:
            with open(Config().main_exec_data_file) as f:
                main_exec_data = json.load(f)
        return main_exec_data

    train_data = dm.run_and_register_to_workers(load_eval_data)
    main_exec_data = dm.run_and_register_to_workers(load_main_exec_data)

    exec = Execution(fl.exec_data_file, main_exec_data)

    counter = 0
    while True:
        print_info(f"Collection {counter}...", end=' ', add_label=False)
        results = dm.map_objs(execute_bench, range(len(train_data)), train_data, main_exec_data, training=False, obj_str=lambda i: train_data[i].bench_name)
        new_cache_data: dict[str, dict[str, int]] = {}
        cache_misses = 0
        for res in results:
            if not res:
                continue
            exec_time, bench_name, cache_key, cache_miss = res
            cache_misses += int(cache_miss)
            if exec_time is None or not cache_miss:
                continue
            if bench_name not in new_cache_data:
                new_cache_data[bench_name] = {}
            new_cache_data[bench_name][cache_key] = exec_time
        exec.update_execution_cache(new_cache_data)
        print_info(f"{cache_misses / len(results) * 100:.2f}% new records")
        counter += 1
        sys.stdout.flush()
        sys.stderr.flush()
