from typing import Optional
from rl_autoschedular.benchmarks import Benchmarks
from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from .singleton import Singleton
from .log import print_info
from .config import Config
import json
import os


class DaskManager(metaclass=Singleton):
    def __init__(self):
        cluster = SLURMCluster(
            job_name='dask',
            queue='compute',
            cores=28,
            processes=1,
            memory='100GB',
            walltime='7-00',
            job_extra_directives=[
                '--reservation=c2',
                '--nodes=1',
                '--exclusive',
            ],
            log_directory='dask-logs',
            job_script_prologue=[
                'module load miniconda-nobashrc',
                'eval "$(conda shell.bash hook)"',
                f'conda activate {os.getenv("CONDA_ENV")}',
                'export OMP_NUM_THREADS=12',
            ],
            scheduler_options={
                'dashboard': False
            }
        )

        num_nodes_to_use = int(os.environ["DASK_NODES"])
        print_info(f"Requesting {num_nodes_to_use} nodes for Dask workers...")
        cluster.scale(jobs=num_nodes_to_use)

        client = Client(cluster)
        print_info(f"Dask client connected! Dashboard at: {client.dashboard_link}")

        self.cluster = cluster
        self.client = client
        self.workers_names = list(cluster.workers.keys())
        self.num_workers = len(cluster.workers)

    def load_train_data(self):
        def __load_train_data(_):
            return Benchmarks()
        self.remote_train_data = []
        for i, worker in enumerate(self.workers_names):
            self.remote_train_data.append(self.client.submit(__load_train_data, i, workers=[worker]))
        return __load_train_data(-1)

    def load_eval_data(self):
        def __load_eval_data(_):
            return Benchmarks(is_training=False)
        self.remote_eval_data = []
        for i, worker in enumerate(self.workers_names):
            self.remote_eval_data.append(self.client.submit(__load_eval_data, i, workers=[worker]))
        return __load_eval_data(-1)

    def load_main_exec_data(self):
        def __load_main_exec_data(_):
            main_exec_data: Optional[dict[str, dict[str, int]]] = None
            if Config().main_exec_data_file:
                with open(Config().main_exec_data_file) as f:
                    main_exec_data = json.load(f)
            return main_exec_data
        self.remote_main_exec_data = []
        for i, worker in enumerate(self.workers_names):
            self.remote_main_exec_data.append(self.client.submit(__load_main_exec_data, i, workers=[worker]))
        return __load_main_exec_data(-1)

    def close(self):
        self.client.close()
        self.cluster.close()
