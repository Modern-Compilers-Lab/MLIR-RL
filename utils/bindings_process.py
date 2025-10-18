import multiprocessing
from typing import Callable, Optional, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing import Queue

T = TypeVar('T')
ENABLED = False
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
        p.join(timeout)
        if p.is_alive():
            p.kill()
            raise TimeoutError(f"Bindings call {func.__name__} timed out")

        if ec := p.exitcode:
            raise Exception(f"Bindings call {func.__name__} failed with exit code: {ec}")

        res = q.get_nowait()
        if isinstance(res, Exception):
            raise res
        return res


def _func_wrapper(q: 'Queue', func: Callable, *args):
    try:
        q.put(func(*args))
    except Exception as e:
        q.put(e)
