import random
import string
import os
from typing import Any, Optional, Literal
import time
import json
from utils.neptune_utils import init_neptune
from utils.config import RASConfig, AASConfig
import neptune


def generate_random_string():
    """Generate a random string of length 10"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))


def print_info(*args):
    """Prints an information message"""
    message = ' '.join(map(str, args))
    print(f"\033[94m[INFO]\t {message}\033[0m")


def print_success(*args):
    """Prints a success message"""
    message = ' '.join(map(str, args))
    print(f"\033[92m[SUCCESS]\t {message}\033[0m")


def print_alert(*args):
    """Prints an alert message"""
    message = ' '.join(map(str, args))
    print(f"\033[93m[ALERT]\t {message}\033[0m")


def print_error(*args):
    """Prints an error message"""
    message = ' '.join(map(str, args))
    print(f"\033[91m[ERROR]\t {message}\033[0m")


class FileLogger:
    """A class to log messages to files."""

    ROOT_DIR = 'output'

    def __init__(self):
        # Create directory name from timestamp
        base_dir_name = time.strftime("%m-%d-%Y-%H-%M-%S")
        self.base_dir_path = os.path.join(FileLogger.ROOT_DIR, base_dir_name)

    def append(self, path: str, message: Any):
        """Log a message.

        Args:
            path (str): The path to the log file.
            message (Any): The message to log.
        """
        # If path or message are not given, skip logging
        if not path or not message:
            return
        # If path does not exist, create it
        full_path = os.path.join(self.base_dir_path, path)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        # Write message in file
        with open(full_path, 'a') as f:
            f.write(str(message) + '\n')

    def extend(self, path: str, messages: list[Any]):
        """Log multiple messages.

        Args:
            path (str): The path to the log file.
            messages (list[Any]): A list of messages to log.
        """
        # If path or messages are not given, skip logging
        if not path or not messages:
            return
        # If path does not exist, create it
        full_path = os.path.join(self.base_dir_path, path)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        # Write messages in file
        with open(full_path, 'a') as f:
            f.write(''.join([str(message) + '\n' for message in messages]))

    def upload_dict(self, path: str, data: dict, indent: Optional[int] = None):
        """Log a dictionary.

        Args:
            path (str): The path to the log file.
            data (dict): The dictionary to log.
        """
        # If path or data are not given, skip logging
        if not path or not data:
            return
        # If path does not exist, create it
        full_path = os.path.join(self.base_dir_path, path)
        if not os.path.exists(os.path.dirname(full_path)):
            os.makedirs(os.path.dirname(full_path))
        # Write data in file
        with open(full_path, 'w') as f:
            json.dump(data, f, indent=indent)


class NeptuneLogger:
    """A class to log messages to Neptune."""

    log_dir_path: str
    """The path to the directory containing log files."""
    run: neptune.Run
    """The Neptune run object."""
    cursors: dict[str, int]
    """Current indexes for each log file."""

    def __init__(self, log_dir_path: str, algorithm: Literal["ras", "aas"] = "ras"):
        self.log_dir_path = log_dir_path
        cfg_path = os.path.join(log_dir_path, 'config.json')
        if algorithm == "ras":
            cfg = RASConfig()
        else:
            cfg = AASConfig()
        cfg.load_from_json(path=cfg_path)
        self.run = init_neptune(cfg.tags, algo=algorithm, cfg_path=cfg_path)
        self.cursors = {}

    def start(self):
        """Start logging to neptune from log files."""
        while True:
            # For each log file
            for log_file_path in self.__get_all_file_paths(self.log_dir_path, exclude=['config.json']):
                # Get relative path
                log_file_rel_path = os.path.relpath(log_file_path, self.log_dir_path)
                # If log file is not in cursors, add it
                if log_file_rel_path not in self.cursors:
                    self.cursors[log_file_rel_path] = 0
                # Read the new part of log file
                with open(log_file_path, 'r') as f:
                    f.seek(self.cursors[log_file_rel_path])
                    s = f.read()
                    lines = s.split('\n')
                    bytes_read = len(s.encode('utf-8'))
                # Get new lines
                data = [float(e.strip()) for e in lines if len(e) > 0]
                # Log new lines
                self.run[log_file_rel_path].extend(data)
                # Update cursor
                self.cursors[log_file_rel_path] += bytes_read
            # Wait for 1 minute
            time.sleep(60)

    def __get_all_file_paths(self, directory: str, exclude: list[str] = []):
        """Get all file paths in a directory.

        Args:
            directory (str): The directory to search.
            exclude (list[str], optional): A list of file names to exclude. Defaults to [].

        Returns:
            Generator[str]: A generator of file paths.
        """
        for dirpath, _, filenames in os.walk(directory):
            for f in filenames:
                if f not in exclude:
                    yield os.path.join(dirpath, f)
