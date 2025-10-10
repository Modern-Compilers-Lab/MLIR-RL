import subprocess
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Optional, TypeVar

from distributed import Future, as_completed
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from .file_logger import FileLogger
from .singleton import Singleton
from .log import print_alert, print_error, print_info, print_success
import os

if TYPE_CHECKING:
    from rl_autoschedular.benchmarks import Benchmarks
    from dask_jobqueue.slurm import SLURMJob
    from rl_autoschedular.state import OperationState

ENABLED = True
T = TypeVar('T')


class DaskManager(metaclass=Singleton):

    def __init__(self):
        if not ENABLED:
            return

        enable_dashboard = True
        cluster = SLURMCluster(
            job_name='dask',
            queue='compute',
            cores=28,
            processes=1,
            nanny=True,
            memory='100GB',
            walltime='7-00',
            job_extra_directives=[
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
                'dashboard': enable_dashboard,
                'worker_ttl': '3600s'
            }
        )
        self.cluster = cluster

        num_nodes_to_use = int(os.environ["DASK_NODES"])
        print_info(f"Requesting {num_nodes_to_use} nodes for Dask workers...")
        cluster.scale(jobs=num_nodes_to_use)
        self.__keep_only_running()
        print_success(f"Got {self.num_workers} nodes")

        client = Client(cluster)
        self.client = client
        print_success("Dask client connected!", f"  Dashboard at: {client.dashboard_link}" if enable_dashboard else "")

        self.batch_timeout = 300
        self.persistent_funcs: dict[str, Callable[[], Any]] = {}
        self.persistent_futures: dict[str, Future] = {}

    @property
    def workers_names(self) -> list[str]:
        if not ENABLED:
            return []
        return list(self.cluster.workers.keys())

    @property
    def num_workers(self) -> int:
        if not ENABLED:
            return 0
        return len(self.cluster.workers)

    def map_states(
        self,
        func: Callable[['OperationState', str, 'Benchmarks', Optional[dict[str, dict[str, int]]]], T],
        states: list['OperationState'],
        benchs: 'Benchmarks',
        main_exec_data: Optional[dict[str, dict[str, int]]],
        training: bool,
    ) -> list[Optional[T]]:
        if not ENABLED or self.num_workers == 0:
            return [func(s, FileLogger().exec_data_file, benchs, main_exec_data) for s in states]

        # Prepare states for submission
        states_count = len(states)
        ordered_states = list(zip(range(states_count), states))
        results: list[Optional[T]] = [None] * states_count
        future_to_worker: dict[Future, str] = {}

        # Submit first states to each worker
        initial_states_count = min(states_count, self.num_workers)
        for i in range(initial_states_count):
            worker_name = self.workers_names[i]
            future = self.__submit_state(func, *ordered_states.pop(0), worker_name, training)
            future_to_worker[future] = worker_name

        # Process futures as they finish
        ac = as_completed(future_to_worker.keys(), with_results=True, timeout=self.batch_timeout if training else None)
        try:
            for future, indexed_result in ac:
                future: Future
                indexed_result: tuple[int, T]

                idx, result = indexed_result
                results[idx] = result
                freed_worker = future_to_worker.pop(future)

                # If there are still remaining states submit them
                if ordered_states:
                    new_future = self.__submit_state(func, *ordered_states.pop(0), freed_worker, training)
                    future_to_worker[new_future] = freed_worker

                    # Include the new future in the queue
                    ac.add(new_future)

        except TimeoutError:
            self.client.cancel(list(future_to_worker.keys()), reason='task-timeout', msg='Task timed out', force=True)
            failed_workers = list(future_to_worker.values())
            try:
                self.client.restart_workers(failed_workers, raise_for_error=False)
            except Exception:
                pass
            restarted_workers = set(failed_workers).intersection(set(self.workers_names))
            unrestarted_workers = set(failed_workers) - set(self.workers_names)
            for worker in restarted_workers:
                self.__renew_worker_persistents(worker)
            print_error(
                "States exec timed out\n"
                f"Cancelling benchmarks: {[s.bench_name for s, r in zip(states, results) if r is None]}\n"
                f"Unvisited benchmarks: {[s.bench_name for _, s in ordered_states]}\n"
                f"Restarted workers: {restarted_workers}\n"
                f"Failed to restart workers: {unrestarted_workers}"
            )

        return results

    def run_and_register_to_workers(self, func: Callable[[], T]):
        if not ENABLED or self.num_workers == 0:
            return func()

        key = func.__name__
        if key in self.persistent_funcs:
            return func()
        self.persistent_funcs[key] = func

        for worker in self.workers_names:
            self.__submit_persistent(key, worker)

        return func()

    def __submit_persistent(self, key: str, worker: str):
        assert key in self.persistent_funcs, f"Task {key} expected to be registered"
        func = self.persistent_funcs[key]

        worker_key = f'{key}_{worker}'
        assert worker_key not in self.persistent_futures, f"Future {key} was found existing in worker {worker}"

        future = self.client.submit(
            func,
            workers=worker,
            pure=False
        )
        self.persistent_futures[worker_key] = future

        return future

    def __get_persistent(self, key: str, worker: str):
        worker_key = f'{key}_{worker}'
        if worker_key in self.persistent_futures:
            return self.persistent_futures[worker_key]

        print_alert(f"Future {key} not found in worker {worker}, attemtping recomputation!")
        if key in self.persistent_funcs:
            return self.__submit_persistent(key, worker)

        raise Exception(f"Unable to find or compute future {key}")

    def __renew_persistent(self, key: str, worker: str):
        worker_key = f'{key}_{worker}'
        if worker_key in self.persistent_futures:
            del self.persistent_futures[worker_key]

        return self.__submit_persistent(key, worker)

    def __renew_worker_persistents(self, worker: str):
        for key in self.persistent_funcs:
            self.__renew_persistent(key, worker)

    def __submit_state(
        self,
        func: Callable[['OperationState', str, 'Benchmarks', Optional[dict[str, dict[str, int]]]], T],
        idx: int,
        state: 'OperationState',
        worker: str,
        training: bool
    ):
        # Add a wrapper to track state order
        def func_wrapper(idx: int, *args):
            return idx, func(*args)
        func_wrapper.__name__ = func.__name__ + '_wrapper'

        exec_data_file = FileLogger().exec_data_file
        benchs = self.__get_persistent('load_train_data' if training else 'load_eval_data', worker)
        main_exec_data = self.__get_persistent('load_main_exec_data', worker)

        return self.client.submit(
            func_wrapper,
            idx, state, exec_data_file, benchs, main_exec_data,
            workers=worker,
            resources={'single_task_slot': 1},
            pure=False
        )

    def __keep_only_running(self):
        """Keep only workers with running jobs"""

        # Wait for the cluster to submit the jobs
        async def _():
            await self.cluster
        self.cluster.sync(_)

        # Get workers and their job ids
        workers: dict[str, 'SLURMJob'] = self.cluster.workers
        if not workers:
            return
        job_id_to_worker = {j.job_id: w for w, j in workers.items() if isinstance(j.job_id, str)}

        # Give it some time for the jobs to be accepted
        sleep(5)

        command = ['squeue', '-h', '-o', '%i %T', '-j', ','.join(job_id_to_worker.keys())]
        running_workers: set[str]
        try:
            # Run the command
            result = subprocess.run(command, capture_output=True, text=True, check=True)

            # The output will be one status per line
            output_statuses = result.stdout.strip().split('\n')

            # Map job IDs to their retrieved statuses
            running_workers = set()
            for id_status in output_statuses:
                job_id, status = id_status.split()
                if status != 'RUNNING':
                    continue
                running_workers.add(job_id_to_worker[job_id])

        except subprocess.CalledProcessError:
            running_workers = set()
        non_running_workers = set(workers.keys()) - running_workers
        if non_running_workers:
            print_alert(
                f"Jobs weren't accepted for workers: {non_running_workers}\n"
                "Removing those workers from cluster"
            )
            self.cluster.sync(self.cluster.scale_down, non_running_workers)

    def close(self):
        self.client.close()
        self.cluster.close()
