from .singleton import Singleton
from .config import Config
import json
import os


class FileLogger(metaclass=Singleton):
    """Class to log results to files"""
    def __init__(self):
        cfg = Config()
        tags = ['ppo'] + cfg.tags

        # Create run dir
        dir_path = cfg.results_dir
        subdir_ids = sorted([int(d.split('_')[-1]) for d in os.listdir(dir_path) if d.startswith('run_')])
        run_id = subdir_ids[-1] + 1 if subdir_ids else 0
        self.run_dir = os.path.join(dir_path, f'run_{run_id}')
        os.makedirs(self.run_dir, exist_ok=True)

        # Create tags file
        tags_file = os.path.join(self.run_dir, 'tags')
        with open(tags_file, 'w') as f:
            f.write('\n'.join(tags))
            f.write('\n')

        # Create exec data file
        self.exec_data_file = os.path.join(self.run_dir, 'exec_data.json')
        with open(self.exec_data_file, "w") as f:
            json.dump({}, f)

        # Create logs dir
        self.logs_dir = os.path.join(self.run_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)

        # Create models dir
        self.models_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        # Init files dict
        self.files_dict: dict[str, FileInstance] = {}

    def __getitem__(self, path: str):
        if path not in self.files_dict:
            full_path = os.path.join(self.logs_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            assert not os.path.exists(full_path), f"File {path} already exists"
            self.files_dict[path] = FileInstance(full_path)
        return self.files_dict[path]


class FileInstance:
    def __init__(self, path: str):
        self.path = path

    def append(self, data):
        with open(self.path, 'a') as f:
            f.write(str(data))
            f.write('\n')

    def extend(self, data: list):
        with open(self.path, 'a') as f:
            f.write('\n'.join(map(str, data)))
            f.write('\n')
