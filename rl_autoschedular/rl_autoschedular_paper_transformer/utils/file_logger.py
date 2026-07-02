"""Result logging and file management for training metrics.

This module provides file-based logging for training metrics, model artifacts,
and execution results. It manages result directories and enables time-series
metric tracking throughout training.
"""

from typing import Optional
from .singleton import Singleton
from .config import Config
import json
import os


class FileLogger(metaclass=Singleton):
    """Class to log results to files"""
    def __init__(self):
        self.enabled = True
        cfg = Config()

        # Write directly to results_dir (no run_N/ subdirectory)
        self.run_dir = cfg.results_dir
        os.makedirs(self.run_dir, exist_ok=True)

        # Create tags file
        tags_file = os.path.join(self.run_dir, 'tags')
        with open(tags_file, 'w') as f:
            f.write('\n'.join(cfg.tags))
            f.write('\n')

        # Create exec data file inside logs/
        self.logs_dir = os.path.join(self.run_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.exec_data_file = os.path.join(self.logs_dir, 'exec_data.json')
        with open(self.exec_data_file, "w") as f:
            json.dump({}, f)

        # Create models dir
        self.models_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Init files dict
        self.files_dict: dict[str, FileInstance] = {}

    def __getitem__(self, path: str):
        if not self.enabled:
            return FileInstance(None)
        if path not in self.files_dict:
            full_path = os.path.join(self.logs_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            self.files_dict[path] = FileInstance(full_path)
        return self.files_dict[path]

    def disable_logging(self):
        self.enabled = False


class FileInstance:
    def __init__(self, path: Optional[str]):
        if path is None:
            path = os.devnull
        self.path = path

    def append(self, data):
        with open(self.path, 'a') as f:
            f.write(str(data))
            f.write('\n')

    def extend(self, data: list):
        with open(self.path, 'a') as f:
            f.write('\n'.join(map(str, data)))
            f.write('\n')
