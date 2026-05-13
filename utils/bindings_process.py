import concurrent.futures
import logging
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Optional, TypeVar

T = TypeVar('T')
logger = logging.getLogger(__name__)

# Use spawn context to avoid fork-safety issues with MLIR C++ bindings.
_ctx = multiprocessing.get_context("spawn")

ENABLED = True

# Restart pool workers every N calls to prevent memory corruption buildup.
_MAX_TASKS_PER_WORKER = 500

_pool: Optional[ProcessPoolExecutor] = None
_pool_call_count = 0


def _get_pool() -> ProcessPoolExecutor:
    """Lazily create (or recreate) a single-worker spawn-based process pool."""
    global _pool, _pool_call_count
    if _pool is None or _pool_call_count >= _MAX_TASKS_PER_WORKER:
        if _pool is not None:
            _pool.shutdown(wait=False)
            logger.debug("Recycling BindingsProcess pool after %d calls", _pool_call_count)
        _pool = ProcessPoolExecutor(max_workers=1, mp_context=_ctx)
        _pool_call_count = 0
    return _pool


def _reset_pool():
    """Force-restart the pool (e.g. after a worker crash)."""
    global _pool, _pool_call_count
    if _pool is not None:
        _pool.shutdown(wait=False)
    _pool = None
    _pool_call_count = 0


class BindingsProcess:
    @staticmethod
    def call(func: Callable[..., T], *args, timeout: Optional[float] = None) -> T:
        if not ENABLED:
            return func(*args)

        global _pool_call_count
        pool = _get_pool()
        _pool_call_count += 1

        future = pool.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except (concurrent.futures.TimeoutError, TimeoutError):
            # `Future.result(timeout=...)` raises `concurrent.futures.TimeoutError`
            # (which on Python ≥ 3.11 *is* the built-in `TimeoutError`). We catch
            # both names defensively. The previous `multiprocessing.context.TimeoutError`
            # catch did not match — letting empty-args TimeoutErrors silently fall
            # through to the generic Exception branch and propagate as `str(e) == ""`.
            timeout_msg = f"Bindings call {func.__name__} timed out after {timeout}s"
            _reset_pool()
        except Exception as e:
            # Check if this was a worker crash (BrokenProcessPool)
            err_msg = str(e)
            if "Broken" in type(e).__name__ or "exit code" in err_msg.lower():
                _reset_pool()
                raise RuntimeError(
                    f"Bindings call {func.__name__} crashed (worker died): {e}"
                ) from e
            raise

        # Raise the new TimeoutError outside the `except` block so Python doesn't
        # implicitly chain the old empty-args TimeoutError via `__context__`.
        raise TimeoutError(timeout_msg)
