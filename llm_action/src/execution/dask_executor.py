import logging
import tempfile
from pathlib import Path

from dask.distributed import Client
from dask_jobqueue import SLURMCluster

from llm_action.src.config import DASK_TMP_DIR, CONDA_ENV, DASK_WAIT_TIMEOUT, DASK_TIMEOUT

_shared_client: Client | None = None
_shared_cluster: SLURMCluster | None = None

logger = logging.getLogger(__name__)

def _execute_torch_on_worker(op_type: str, dims: tuple[int, ...]) -> float:
    """Run on a Dask worker: call the matching execute_torch_* function in-process.

    Args are tiny ints so no temp-file dance is needed (unlike _execute_on_worker
    which writes MLIR code to a shared file to avoid the scheduler pickle hop).
    Returns median execution time in milliseconds.
    """
    from llm_action.src.execution.torch_execution import (
        execute_torch_matmul, execute_torch_conv2d, execute_torch_add,
        execute_torch_pooling_nchw_max, execute_torch_relu,
    )
    if op_type == "matmul":
        return execute_torch_matmul(*dims)
    if op_type == "conv2d":
        return execute_torch_conv2d(*dims)
    if op_type == "add":
        return execute_torch_add(*dims)
    if op_type == "pooling_nchw_max":
        return execute_torch_pooling_nchw_max(*dims)
    if op_type == "relu":
        return execute_torch_relu(tuple(dims))
    raise ValueError(f"Unknown torch op_type: {op_type}")

def _execute_on_worker(code_path: str) -> tuple[float, bool]:
    """Run on a Dask worker: hand the on-disk MLIR path straight through to
    the BindingsProcess spawn child and clean up the temp file in `finally`.

    The spawn child reads the file itself, skipping the worker→spawn-child
    pickle of the 10-100 KB code string, and runs bufferize+execute in a
    single round-trip (the bufferized intermediate stays in spawn-child
    memory). Saves two large pickles per call versus the in-memory variant.

    The Dask worker is launched non-daemonic (`nanny=False` in the
    SLURMCluster config) so that BindingsProcess can spawn an isolation
    subprocess. MLIR/LLVM SIGSEGVs on bad IR stay inside that spawn child —
    the Dask worker survives, BindingsProcess returns a clean RuntimeError,
    and the next task runs. Cluster-level recovery on rare actual worker
    death is handled by `cluster.adapt(...)`.

    Returns (execution_time_ms, success).
    """
    from pathlib import Path

    from llm_action.src.execution.mlir_execution import execute_mlir_from_path

    try:
        exec_time_ns, success = execute_mlir_from_path(code_path)
    finally:
        Path(code_path).unlink(missing_ok=True)
    return exec_time_ns / 1_000_000, success

class DaskExecutor:
    """Drop-in replacement for SlurmExecutor using a shared Dask cluster."""

    def __init__(self, client: Client, timeout: int = DASK_TIMEOUT):
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

    def execute_torch(self, op_type: str, dims: tuple[int, ...]) -> float:
        future = self.client.submit(_execute_torch_on_worker, op_type, tuple(dims), pure=False)
        try:
            return future.result(timeout=self.timeout)
        except Exception:
            future.cancel()
            raise

def create_dask_client(
    n_workers: int,
    conda_env: str = CONDA_ENV,
    wait_timeout: int = DASK_WAIT_TIMEOUT,
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
        # Non-daemonic worker: lets BindingsProcess spawn an isolation
        # subprocess inside the worker. With nanny=True the worker would be
        # daemonic and Python's multiprocessing would reject any child
        # process creation. Loss of the Nanny's auto-restart-on-crash and
        # memory watcher is intentional: subprocess isolation absorbs MLIR
        # crashes, and `cluster.adapt(...)` below provides cluster-level
        # respawn for any real worker-level failure.
        nanny=False,
        memory="100GB",
        walltime="5-00",
        job_extra_directives=[
            "--reservation=c2",
            "--qos=c2",
            "--nodes=1",
            "--exclusive",
        ],
        worker_extra_args=["--resources", "single_task_slot=1"],
        log_directory="dask-logs",
        job_script_prologue=[
            "module load miniconda-nobashrc",
            'eval "$(conda shell.bash hook)"',
            f"conda activate {conda_env}",
            "export OMP_NUM_THREADS=28",
            "export OMP_PROC_BIND=close",
            "export OMP_PLACES=cores",
            "export OMP_SCHEDULE=static",
            "export OMP_DYNAMIC=FALSE",
            "export OMP_WAIT_POLICY=passive",
            "export KMP_BLOCKTIME=0",
        ],
        scheduler_options={"dashboard": True},
    )

    logger.info(f"Requesting {n_workers} Dask worker nodes (adaptive)...")
    cluster.adapt(minimum=n_workers, maximum=n_workers)
    client = Client(cluster)
    logger.info(f"Waiting for {n_workers} workers (timeout={wait_timeout}s)...")
    client.wait_for_workers(n_workers, timeout=wait_timeout)
    logger.info(
        f"Dask cluster ready (adaptive, target={n_workers}): "
        f"{len(client.scheduler_info()['workers'])} workers | "
        f"dashboard: {client.dashboard_link}"
    )
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
