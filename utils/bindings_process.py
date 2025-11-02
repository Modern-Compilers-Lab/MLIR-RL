import multiprocessing
from multiprocessing.connection import wait
import os
import queue
import signal
from typing import Callable, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing import Queue

T = TypeVar('T')
ENABLED = os.getenv('ENABLE_BINDINGS_PROCESS', '0') == '1'
ENABLE_TIMEOUT = False


class BindingsProcess:
    @staticmethod
    def call(func: Callable[..., T], *args, timeout: Optional[float] = None) -> T:
        if not ENABLED:
            return func(*args)
        if not ENABLE_TIMEOUT:
            timeout = None

        ctx = multiprocessing.get_context('fork')
        q = ctx.Queue()
        p = ctx.Process(target=_func_wrapper, args=(q, func, *args), daemon=True)
        p.start()
        ready = wait([p.sentinel, q._reader.fileno()], timeout=timeout)
        if not ready:
            p.kill()
            p.join()
            raise TimeoutError(f"Bindings call {func.__name__} timed out")

        try:
            res = q.get_nowait()
            p.join()
            if isinstance(res, Exception):
                raise res
            return res
        except queue.Empty:
            p.join()
            ec = p.exitcode
            msg = f"Bindings call {func.__name__} failed"

            if ec and ec < 0:
                try:
                    signame = signal.Signals(-ec).name
                    msg += f" with signal: {signame} (exit code: {ec})"
                except ValueError:
                    msg += f" with exit code: {ec}"
            else:
                msg += f" with exit code: {ec}"

            raise Exception(msg)


def _func_wrapper(q: 'Queue', func: Callable, *args):
    try:
        res = func(*args)
        q.put(res)
    except Exception as e:
        q.put(e)
