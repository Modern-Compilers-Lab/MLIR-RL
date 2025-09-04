from multiprocessing import Process, Queue
from typing import Callable, Optional, TypeVar

T = TypeVar('T')
ENABLED = False


class BindingsProcess:
    @staticmethod
    def call(func: Callable[..., T], *args, timeout: Optional[float] = None) -> T:
        if not ENABLED:
            return func(*args)

        def func_wrapper(q: Queue):
            try:
                q.put(func(*args))
            except Exception as e:
                q.put(e)

        q = Queue()
        p = Process(target=func_wrapper, args=(q,), daemon=True)
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
