import logging
import tempfile
from pathlib import Path

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from llm_action.src.config import DASK_TMP_DIR, CONDA_ENV, DASK_WAIT_TIMEOUT

_shared_client: Client | None = None
_shared_cluster: SLURMCluster | None = None

logger = logging.getLogger(__name__)

def _execute_on_worker(code_path: str) -> tuple[float, bool]:
    """Run on a Dask worker: read MLIR code from shared filesystem, execute.

    Dask workers already provide process isolation, so BindingsProcess
    subprocess spawning is disabled here to avoid double-isolation overhead.

    Returns (execution_time_ms, success).
    """
    from pathlib import Path
    import utils.bindings_process as bp
    bp.ENABLED = False

    from llm_action.src.execution.mlir_execution import execute_mlir

    code = Path(code_path).read_text()
    Path(code_path).unlink(missing_ok=True)
    exec_time_ns, success = execute_mlir(code)
    return exec_time_ns / 1_000_000, success

class DaskExecutor:
    """Drop-in replacement for SlurmExecutor using a shared Dask cluster."""

    def __init__(self, client: Client, timeout: int = 120):
        self.client = client
        self.timeout = timeout

    def execute(self, code: str) -> tuple[float, bool]:
        # Write code to shared filesystem to avoid sending large payloads
        # through the Dask scheduler (MLIR code can grow to 10+ MiB).
        DASK_TMP_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", dir=DASK_TMP_DIR, delete=False
        ) as f:
            f.write(code)
            code_path = f.name

        future = self.client.submit(_execute_on_worker, code_path, pure=False)
        try:
            return future.result(timeout=self.timeout)
        except Exception:
            future.cancel()
            Path(code_path).unlink(missing_ok=True)
            raise

def create_dask_client(
    n_workers: int,
    conda_env: str = "mlir",
    wait_timeout: int = 300,
) -> tuple[SLURMCluster, Client]:
    """Create a SLURMCluster and Client with persistent workers.

    Mirrors the cluster config from utils/dask_manager.py but without the
    utils.keys dependency chain (NEPTUNE_PROJECT, etc.).
    """
    cluster = SLURMCluster(
        job_name="dask",
        queue="compute",
        cores=28,
        processes=1,
        nanny=True,
        memory="100GB",
        walltime="5-00",
        job_extra_directives=[
            "--reservation=c2",
            "--nodes=1",
            "--exclusive",
        ],
        worker_extra_args=["--resources", "single_task_slot=1"],
        log_directory="dask-logs",
        job_script_prologue=[
            "module load miniconda-nobashrc",
            'eval "$(conda shell.bash hook)"',
            f"conda activate {conda_env}",
            "export OMP_NUM_THREADS=12",
            "export OMP_PROC_BIND=close",
            "export OMP_PLACES=cores",
            "export OMP_SCHEDULE=static",
            "export OMP_DYNAMIC=FALSE",
            "export OMP_WAIT_POLICY=passive",
            "export KMP_BLOCKTIME=0",
        ],
        scheduler_options={"dashboard": True, "worker_ttl": "3600s"},
    )

    logger.info(f"Requesting {n_workers} Dask worker nodes...")
    cluster.scale(jobs=n_workers)
    client = Client(cluster)
    logger.info(f"Waiting for {n_workers} workers (timeout={wait_timeout}s)...")
    client.wait_for_workers(n_workers, timeout=wait_timeout)
    logger.info(f"Dask cluster ready: {len(client.scheduler_info()['workers'])} workers | dashboard: {client.dashboard_link}")
    return cluster, client

def init_shared_client(n_workers: int, conda_env: str = CONDA_ENV, wait_timeout: int = DASK_WAIT_TIMEOUT):
    """Initialize the module-level shared Dask client (call once before env creation)."""
    global _shared_client, _shared_cluster
    if _shared_client is not None:
        return
    _shared_cluster, _shared_client = create_dask_client(n_workers, conda_env, wait_timeout)

def get_shared_client() -> Client:
    """Get the shared Dask client. Raises if not initialized."""
    if _shared_client is None:
        raise RuntimeError("Dask client not initialized. Call init_shared_client() first.")
    return _shared_client

def close_shared_client():
    """Shut down the shared Dask cluster and client."""
    global _shared_client, _shared_cluster
    if _shared_client is not None:
        _shared_client.close()
        _shared_client = None
    if _shared_cluster is not None:
        _shared_cluster.close()
        _shared_cluster = None
