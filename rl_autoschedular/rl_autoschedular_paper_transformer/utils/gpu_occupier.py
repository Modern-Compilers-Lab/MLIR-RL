"""GPU resource management and memory occupancy utilities.

This module provides utilities for managing GPU resources during training,
including methods to reserve GPU memory and ensure consistent GPU availability
for model training.
"""

from contextlib import contextmanager
import multiprocessing
from typing import TYPE_CHECKING

from .log import print_alert
from .singleton import Singleton

if TYPE_CHECKING:
    from multiprocessing import Process
    from multiprocessing.context import SpawnContext
    from multiprocessing.synchronize import Event
    from torch import device as Device


MATRIX_SIZE = 64


class GPUOccupier(metaclass=Singleton):
    """Manages a parallel process to keep the GPU busy when it is idle."""

    __ctx: 'SpawnContext'
    """Multiprocessing context."""

    __process: 'Process'
    """Process that keeps the GPU busy."""

    __gpu_needed_event: 'Event'
    """Event that is set when the GPU is needed."""

    __stop_event: 'Event'
    """Event that is set when the process should stop."""

    def __init__(self):
        self.__ctx = multiprocessing.get_context('spawn')
        self.__gpu_needed_event = self.__ctx.Event()
        self.__stop_event = self.__ctx.Event()
        self.__process = None

    def start(self, device: 'Device'):
        if device.type != 'cuda':
            raise ValueError('GPUOccupier is only for CUDA devices.')

        if self.__process and self.__process.is_alive():
            print_alert("[GPUOccupier] Process already started.")
            return

        self.__process = self.__ctx.Process(
            target=_gpu_occupier_run,
            args=(device, self.__stop_event, self.__gpu_needed_event),
            daemon=True
        )
        self.__process.start()

    @contextmanager
    def gpu_needed(self):
        """Context manager that signals that the GPU is needed."""

        if self.__gpu_needed_event.is_set():
            yield
            return
        self.__gpu_needed_event.set()
        try:
            yield
        finally:
            self.__gpu_needed_event.clear()

    def stop(self):
        self.__stop_event.set()
        if self.__process and self.__process.is_alive():
            self.__process.join(5)
            if self.__process.is_alive():
                self.__process.terminate()
            self.__process = None


def _gpu_occupier_run(device: 'Device', stop_event: 'Event', gpu_needed_event: 'Event'):
    import torch
    from time import sleep
    from rl_autoschedular_paper_transformer.utils.log import print_error, print_info

    print_info("[GPUOccupier] Process started.", flush=True)

    a = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device=device)
    b = torch.randn((MATRIX_SIZE, MATRIX_SIZE), device=device)

    while not stop_event.is_set():
        if not gpu_needed_event.is_set():
            try:
                torch.matmul(a, b)
            except Exception as e:
                print_error("[GPUOccupier] Error:", e, flush=True)
                sleep(5)
        else:
            sleep(1)

    print_info("[GPUOccupier] Process terminating.", flush=True)
