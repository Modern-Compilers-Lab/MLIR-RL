"""Distributed computation management using Dask.

This module handles distributed parallel execution of benchmark evaluations
across multiple worker nodes. It provides abstractions for mapping functions
across data in a distributed manner with resource management.
"""

import os
import subprocess
from time import sleep, time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, TypeVar

from .file_logger import FileLogger
from .singleton import Singleton
from .log import print_alert, print_error, print_info, print_success
from .bindings_process import ENABLED as BP_ENABLED

if TYPE_CHECKING:
    from rl_autoschedular_paper_transformer.benchmarks import Benchmarks
    from distributed import Future

ENABLED = int(os.getenv('DASK_NODES', '0')) > 0
T = TypeVar('T')
obj_T = TypeVar('obj_T')


class DaskManager(metaclass=Singleton):
    """DaskManager class for distributed parallel execution."""

    def __init__(self):
        if not ENABLED:
            return

        from dask_jobqueue import SLURMCluster
        from distributed import Client

        enable_dashboard = True
        dask_reservation = os.getenv('DASK_RESERVATION')
        dask_conda_env = os.getenv('CONDA_ENV')
        cluster = SLURMCluster(
            job_name='dask',
            queue='compute',
            cores=28,
            processes=1,
            nanny=True,
            memory='100GB',
            walltime='7-00',
            job_extra_directives=[
                f'--reservation={dask_reservation}' if dask_reservation else '',
                '--nodes=1',
                '--exclusive',
            ],
            worker_extra_args=['--resources', 'single_task_slot=1'],
            log_directory='dask-logs',
            job_script_prologue=[
                'module load miniconda-nobashrc',
                'eval "$(conda shell.bash hook)"',
                f'conda activate {dask_conda_env}' if dask_conda_env else '',
                'export OMP_NUM_THREADS=12',
                'export DASK_DISTRIBUTED__WORKER__DAEMON=False' if BP_ENABLED else '',
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
        self.persistent_futures: dict[str, 'Future'] = {}

    @property
    def workers_names(self) -> list[str]:
        """List of available worker names."""
        if not ENABLED:
            return []
        return list(self.cluster.workers.keys())

    @property
    def num_workers(self) -> int:
        """Number of available workers."""
        if not ENABLED:
            return 0
        return len(self.cluster.workers)

    def map_objs(
        self,
        func: Callable[[obj_T, str, 'Benchmarks', Optional[dict[str, dict[str, int]]]], T],
        objs: Iterable[obj_T],
        benchs: 'Benchmarks',
        main_exec_data: Optional[dict[str, dict[str, int]]],
        training: bool,
        obj_str: Callable[[obj_T], str] = lambda o: str(o)
    ) -> list[Optional[T]]:
        """Map a function across objects in a distributed manner.

        Args:
            func: The function to apply to each object.
            objs: The objects to apply the function to.
            benchs: The benchmark suite to use.
            main_exec_data: The main execution data (if available).
            training: Whether the mapping is for training. if True,
                the function will be executed with a timeout and the
                training benchmarks will be used instead of evaluation.
            obj_str: A function to convert each object to a string for logging.

        Returns:
            A list of the results of the function applied to each object.
        """

        if not ENABLED or self.num_workers == 0:
            n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
            if n_workers <= 1:
                return [func(o, FileLogger().exec_data_file, benchs, main_exec_data) for o in objs]
            import concurrent.futures
            exec_file = FileLogger().exec_data_file
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(func, o, exec_file, benchs, main_exec_data) for o in objs]
                return [f.result() for f in futures]

        from distributed import as_completed

        # Prepare objs for submission
        objs_count = len(objs)
        ordered_objs = list(zip(range(objs_count), objs))
        results: list[Optional[T]] = [None] * objs_count
        future_to_worker: dict['Future', str] = {}

        # Submit first objs to each worker
        initial_objs_count = min(objs_count, self.num_workers)
        for i in range(initial_objs_count):
            worker_name = self.workers_names[i]
            future = self.__submit_obj(func, *ordered_objs.pop(0), worker_name, training)
            future_to_worker[future] = worker_name

        # Process futures as they finish
        ac = as_completed(future_to_worker.keys(), with_results=True, timeout=self.batch_timeout if training else None)
        try:
            for future, indexed_result in ac:
                future: 'Future'
                indexed_result: tuple[int, T]

                idx, result = indexed_result
                results[idx] = result
                freed_worker = future_to_worker.pop(future)

                # If there are still remaining objs submit them
                if ordered_objs:
                    new_future = self.__submit_obj(func, *ordered_objs.pop(0), freed_worker, training)
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
                f"Cancelling benchmarks: {[obj_str(o) for o, r in zip(objs, results) if r is None]}\n"
                f"Unvisited benchmarks: {[obj_str(o) for _, o in ordered_objs]}\n"
                f"Restarted workers: {restarted_workers}\n"
                f"Failed to restart workers: {unrestarted_workers}"
            )

        return results

    def run_and_register_to_workers(self, func: Callable[[], T]) -> T:
        """Run a function both locally and on the workers.
        The result will be registered to all workers, and
        returned by this function.

        Args:
            func: The function to run.

        Returns:
            The result of the function.
        """

        if not ENABLED or self.num_workers == 0:
            return func()

        key = func.__name__
        if key in self.persistent_funcs:
            return func()
        self.persistent_funcs[key] = func

        for worker in self.workers_names:
            self.__submit_persistent(key, worker)

        return func()

    def __submit_persistent(self, key: str, worker: str) -> 'Future':
        """Submit a persistent function to a worker,
        and keep track of its result (Future) for re-use.

        Args:
            key: The key of the function.
            worker: The worker to submit the function to.

        Returns:
            The future of the function.
        """
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

    def __get_persistent(self, key: str, worker: str) -> 'Future':
        """Get the result of a persistent function from a worker.

        Args:
            key: The key of the function.
            worker: The worker to get the result from.

        Returns:
            The future that points to the result of the function.
        """

        worker_key = f'{key}_{worker}'
        if worker_key in self.persistent_futures:
            return self.persistent_futures[worker_key]

        print_alert(f"Future {key} not found in worker {worker}, attemtping recomputation!")
        if key in self.persistent_funcs:
            return self.__submit_persistent(key, worker)

        raise Exception(f"Unable to find or compute future {key}")

    def __renew_persistent(self, key: str, worker: str) -> 'Future':
        """Recompute the result of a persistent function on a worker.
        This should be called when a persistent result (Future) has
        become invalid (due to a worker failure mostly).

        Args:
            key: The key of the function.
            worker: The worker to renew the result on.

        Returns:
            The future that points to the result of the function.
        """
        worker_key = f'{key}_{worker}'
        if worker_key in self.persistent_futures:
            del self.persistent_futures[worker_key]

        return self.__submit_persistent(key, worker)

    def __renew_worker_persistents(self, worker: str):
        """Recompute all persistent functions on a worker.
        This should be called when a worker has failed.

        Args:
            worker: The worker to renew the results on.
        """
        for key in self.persistent_funcs:
            self.__renew_persistent(key, worker)

    def __submit_obj(
        self,
        func: Callable[[obj_T, str, 'Benchmarks', Optional[dict[str, dict[str, int]]]], T],
        idx: int,
        obj: obj_T,
        worker: str,
        training: bool
    ) -> 'Future':
        """Execute a function on an object, and submit it to a worker.

        Args:
            func: The function to execute.
            idx: The index of the object (for tracking purposes).
            obj: The object to execute the function on.
            worker: The worker to submit the result to.
            training: Whether the object is for training. if True,
                the function will be executed with a timeout and the
                training benchmarks will be used instead of evaluation.

        Returns:
            The future that points to the result of the function.
        """
        # Add a wrapper to track state order
        def func_wrapper(idx: int, *args):
            return idx, func(*args)
        func_wrapper.__name__ = func.__name__ + '_wrapper'

        exec_data_file = FileLogger().exec_data_file
        benchs = self.__get_persistent('load_train_data' if training else 'load_eval_data', worker)
        main_exec_data = self.__get_persistent('load_main_exec_data', worker)

        return self.client.submit(
            func_wrapper,
            idx, obj, exec_data_file, benchs, main_exec_data,
            workers=worker,
            resources={'single_task_slot': 1},
            pure=False
        )

    def __keep_only_running(self):
        """Keep only workers with running jobs"""
        if TYPE_CHECKING:
            from dask_jobqueue.slurm import SLURMJob

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
        pending_jobs = set(job_id_to_worker.keys())
        start_wait = time()
        print_info("Waiting for jobs to be accepted")
        while time() - start_wait < 60 and pending_jobs:
            command = ['squeue', '-h', '-o', '%i %T', '-j', ','.join(job_id_to_worker.keys())]
            running_workers: set[str] = set()
            pending_jobs: set[str] = set()
            try:
                # Run the command
                result = subprocess.run(command, capture_output=True, text=True, check=True)

                # The output will be one status per line
                output_statuses = result.stdout.strip().split('\n')

                # Map job IDs to their retrieved statuses
                for id_status in output_statuses:
                    job_id, status = id_status.split()
                    if status == 'RUNNING':
                        running_workers.add(job_id_to_worker[job_id])
                    elif status == 'PENDING':
                        pending_jobs.add(job_id_to_worker[job_id])
            except subprocess.CalledProcessError:
                pass
            sleep(1)
        non_running_workers = set(workers.keys()) - running_workers
        if non_running_workers:
            print_alert(
                f"Jobs weren't accepted for workers: {non_running_workers}\n"
                "Removing those workers from cluster"
            )
            self.cluster.sync(self.cluster.scale_down, non_running_workers)

    def close(self):
        """Close the cluster and client."""
        self.client.close()
        self.cluster.close()
