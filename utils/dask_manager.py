from typing import TYPE_CHECKING, Callable, Optional, TypeVar

from typeguard import check_type
from distributed import Future, as_completed, get_worker, progress
from rl_autoschedular.benchmarks import Benchmarks
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from utils.file_logger import FileLogger
from .singleton import Singleton
from .log import print_info
from .config import Config
import json
import os

if TYPE_CHECKING:
    from rl_autoschedular.state import OperationState

T = TypeVar('T')


class DaskManager(metaclass=Singleton):
    def __init__(self):
        enable_dashboard = Config().debug
        cluster = SLURMCluster(
            job_name='dask',
            queue='compute',
            cores=28,
            processes=1,
            nanny=False,
            memory='100GB',
            walltime='7-00',
            job_extra_directives=[
                '--reservation=c2',
                '--nodes=1',
                '--exclusive',
            ],
            worker_extra_args=['--resources', 'single_task_slot=1'],
            log_directory='dask-logs',
            job_script_prologue=[
                'module load miniconda-nobashrc',
                'eval "$(conda shell.bash hook)"',
                f'conda activate {os.getenv("CONDA_ENV")}',
                'export OMP_NUM_THREADS=12',
            ],
            scheduler_options={
                'dashboard': enable_dashboard
            }
        )

        num_nodes_to_use = int(os.environ["DASK_NODES"])
        print_info(f"Requesting {num_nodes_to_use} nodes for Dask workers...")
        cluster.scale(jobs=num_nodes_to_use)

        client = Client(cluster)
        print_info("Dask client connected!", f"  Dashboard at: {client.dashboard_link}" if enable_dashboard else "")

        self.cluster = cluster
        self.client = client
        self.workers_names: list[str] = list(cluster.workers.keys())
        self.num_workers = len(cluster.workers)

    def map_states(
        self,
        func: Callable[['OperationState', str, 'Benchmarks', Optional[dict[str, dict[str, int]]]], T],
        states: list['OperationState'],
        training: bool,
    ) -> list[T]:
        # Provide worker data to the function via a wrapper
        def func_wrapper(s: 'OperationState', e: str, idx: int):
            worker = get_worker()
            benchs = check_type(worker.data[f'__load_train_data_{worker.name}' if training else f'__load_eval_data_{worker.name}'], Benchmarks)
            main_exec_data = check_type(worker.data[f'__load_main_exec_data_{worker.name}'], Optional[dict[str, dict[str, int]]])
            return idx, func(s, e, benchs, main_exec_data)
        func_wrapper.__name__ = func.__name__ + '_wrapper'

        # Prepare states for submission
        states_count = len(states)
        ordered_states = list(zip(range(states_count), states))
        results: list[T] = [None] * states_count
        future_to_worker: dict[Future, str] = {}

        # Submit first states to each worker
        initial_states_count = min(states_count, self.num_workers)
        for i in range(initial_states_count):
            worker_name = self.workers_names[i]
            idx, state = ordered_states.pop(0)
            future = self.client.submit(
                func_wrapper,
                state, FileLogger().exec_data_file, idx,
                workers=worker_name,
                resources={'single_task_slot': 1}
            )
            future_to_worker[future] = worker_name

        # Process futures as they finish
        ac = as_completed(future_to_worker.keys(), with_results=True)
        for future, indexed_result in ac:
            future: Future
            indexed_result: tuple[int, T]

            idx, result = indexed_result
            results[idx] = result
            freed_worker = future_to_worker.pop(future)

            # If there are still remaining states submit them
            if ordered_states:
                new_idx, new_state = ordered_states.pop(0)
                new_future = self.client.submit(
                    func_wrapper,
                    new_state, FileLogger().exec_data_file, new_idx,
                    workers=freed_worker,
                    resources={'single_task_slot': 1}
                )
                future_to_worker[new_future] = freed_worker

                # Include the new future in the queue
                ac.add(new_future)

        return results

    def load_train_data(self):
        def __load_train_data():
            return Benchmarks()
        self.remote_train_data: list[Future] = []
        for worker_name in self.workers_names:
            self.remote_train_data.append(self.client.submit(__load_train_data, workers=worker_name, key=f'__load_train_data_{worker_name}'))
        print_info("Loading train benchmarks to workers...")
        progress(self.remote_train_data)
        return __load_train_data()

    def load_eval_data(self):
        def __load_eval_data():
            return Benchmarks(is_training=False)
        self.remote_eval_data: list[Future] = []
        for worker_name in self.workers_names:
            self.remote_eval_data.append(self.client.submit(__load_eval_data, workers=worker_name, key=f'__load_eval_data_{worker_name}'))
        print_info("Loading eval benchmarks to workers...")
        progress(self.remote_eval_data)
        return __load_eval_data()

    def load_main_exec_data(self):
        def __load_main_exec_data():
            main_exec_data: Optional[dict[str, dict[str, int]]] = None
            if Config().main_exec_data_file:
                with open(Config().main_exec_data_file) as f:
                    main_exec_data = json.load(f)
            return main_exec_data
        self.remote_main_exec_data: list[Future] = []
        for worker_name in self.workers_names:
            self.remote_main_exec_data.append(self.client.submit(__load_main_exec_data, workers=worker_name, key=f'__load_main_exec_data_{worker_name}'))
        print_info("Loading main exec data to workers...")
        progress(self.remote_main_exec_data)
        return __load_main_exec_data()

    def close(self):
        self.client.close()
        self.cluster.close()
