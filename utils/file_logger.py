from utils.singleton import Singleton
from utils.config import Config
import os


class FileLogger(metaclass=Singleton):
    """Class to log results to files"""
    def __init__(self):
        cfg = Config()
        dir_path = cfg.results_dir
        tags = ['ppo'] + cfg.tags
        subdir_ids = sorted([int(d.split('_')[-1]) for d in os.listdir(dir_path) if d.startswith('run_')])
        self.run_id = subdir_ids[-1] + 1 if subdir_ids else 0
        self.run_dir = os.path.join(dir_path, f'run_{self.run_id}')
        os.makedirs(self.run_dir, exist_ok=True)
        self.models_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)
        with open(os.path.join(self.run_dir, 'tags'), 'w') as f:
            f.write('\n'.join(tags))
            f.write('\n')
        self.files_dict: dict[str, FileInstance] = {}

    def __getitem__(self, path: str):
        assert path != 'tags', "Cannot access tags file this way"
        assert not path.startswith('models/'), "Models directory is reserved to torch models"
        if path not in self.files_dict:
            full_path = os.path.join(self.run_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            assert not os.path.exists(full_path), f"File {path} already exists"
            self.files_dict[path] = FileInstance(full_path)
        return self.files_dict[path]

    @property
    def tags(self):
        return self.files_dict['tags']


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
