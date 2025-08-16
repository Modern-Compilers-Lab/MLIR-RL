from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from .log import print_info
from .singleton import Singleton
from typing import TYPE_CHECKING
import os

if TYPE_CHECKING:
    from rl_autoschedular.benchmarks import Benchmarks


class DaskManager(metaclass=Singleton):
    def __init__(self):
        cluster = SLURMCluster(
            job_name='dask',
            queue='compute',
            account='xpress',
            cores=28,
            processes=1,
            memory='64GB',
            walltime='7-00',
            job_extra_directives=[
                '--reservation=scomputer-dalma',
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
            # scheduler_options={
            #     'dashboard': False
            # }
        )

        num_nodes_to_use = 16
        print_info(f"Requesting {num_nodes_to_use} nodes for Dask workers...")
        cluster.scale(jobs=num_nodes_to_use)

        client = Client(cluster)
        print_info(f"Dask client connected! Dashboard at: {client.dashboard_link}")

        self.cluster = cluster
        self.client = client
        self.workers_names = list(cluster.workers.keys())
        self.num_workers = len(cluster.workers)

    def load_train_data(self, benchs: 'Benchmarks'):
        self.remote_train_data = self.client.scatter(benchs, broadcast=True)
        return benchs

    def load_eval_data(self, benchs: 'Benchmarks') -> int:
        self.remote_eval_data = self.client.scatter(benchs, broadcast=True)
        return benchs

    def close(self):
        self.client.close()
        self.cluster.close()
