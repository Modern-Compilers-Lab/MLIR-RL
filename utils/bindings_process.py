"""Process management for safe execution of MLIR bindings with timeouts.

This module provides utilities for executing code in isolated processes to prevent
crashes from unstable C++ bindings. It enables timeout management and proper
resource cleanup for MLIR operations.
"""

import multiprocessing
from multiprocessing.connection import wait
import os
import queue
import signal
from typing import Callable, Optional, TypeVar, Any, TYPE_CHECKING
from mlir._mlir_libs._mlir.ir import Module, Context  # type: ignore

if TYPE_CHECKING:
    from multiprocessing import Queue

T = TypeVar('T')
ENABLED = os.getenv('ENABLE_BINDINGS_PROCESS', '0') == '1'
ENABLE_TIMEOUT = False


class BindingsProcess:
    @staticmethod
    def call(func: Callable[..., T], module: Module, *args, timeout: Optional[float] = None, read_only: bool = True) -> T:
        if not ENABLED:
            return func(module, *args)
        if not read_only:
            # NOTE: No support for modifying modules for now
            return func(module, *args)
        if not ENABLE_TIMEOUT:
            timeout = None

        ctx = multiprocessing.get_context('fork')
        q: 'Queue[dict[str, Any]]' = ctx.Queue()
        p = ctx.Process(target=_func_wrapper, args=(q, func, str(module), *args), daemon=True)
        p.start()
        ready = wait([p.sentinel, q._reader.fileno()], timeout=timeout)
        if not ready:
            p.kill()
            p.join()
            raise TimeoutError(f"Bindings call {func.__name__} timed out")

        try:
            res = q.get_nowait()
            p.join()
            if 'exception' in res:
                raise res['exception']
            return res['result']
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


def _func_wrapper(q: 'Queue', func: Callable, code: str, *args):
    try:
        module = Module.parse(code, Context())
        res = func(module, *args)
        q.put({"result": res})
    except Exception as e:
        q.put({"exception": e})
