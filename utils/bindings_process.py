import os
import sys
import base64
import subprocess
import dill
from typing import Callable, Optional, TypeVar

T = TypeVar('T')
ENABLED = False
TRANSFORM_ENABLED = False


class BindingsProcess:
    @staticmethod
    def call(func: Callable[..., T], *args, timeout: Optional[float] = None, enabled: Optional[bool] = None) -> T:
        if enabled is None:
            enabled = ENABLED
        if not enabled:
            return func(*args)

        serialized = base64.b64encode(dill.dumps((func, args))).decode('ascii')
        script = (
            "import dill, base64, os, sys, io\n"
            "func, args = dill.loads(base64.b64decode(" + repr(serialized) + "))\n"
            "old_stdout, old_stderr = sys.stdout, sys.stderr\n"
            "sys.stdout = sys.stderr = open(os.devnull, 'w')\n"
            "try:\n"
            "    result = func(*args)\n"
            "    sys.stdout, sys.stderr = old_stdout, old_stderr\n"
            "    sys.stdout.write('__BINDINGS_OK__' + repr(result) + '\\n')\n"
            "except Exception as e:\n"
            "    sys.stdout, sys.stderr = old_stdout, old_stderr\n"
            "    sys.stdout.write('__BINDINGS_ERR__' + repr(e) + '\\n')\n"
        )

        proc = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, 'PYTHONUNBUFFERED': '1'},
        )

        stdout = proc.stdout.strip()
        if proc.returncode != 0:
            raise Exception(
                f"Bindings call {getattr(func, '__name__', str(func))} failed with exit code {proc.returncode}\n"
                f"stderr: {proc.stderr[:500]}"
            )

        if stdout.startswith('__BINDINGS_OK__'):
            return eval(stdout[len('__BINDINGS_OK__'):])
        elif stdout.startswith('__BINDINGS_ERR__'):
            raise Exception(
                f"Bindings call error: {stdout[len('__BINDINGS_ERR__'):]}\n"
                f"stderr: {proc.stderr[:200]}"
            )
        else:
            raise Exception(
                f"Bindings call unexpected output: {stdout[:200]}\n"
                f"stderr: {proc.stderr[:200]}"
            )
