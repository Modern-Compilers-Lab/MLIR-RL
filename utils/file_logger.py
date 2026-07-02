from .singleton import Singleton
from .config import Config
from .implementation import get_agent_runs_root, get_autoschedular_impl
import json
import os


class FileLogger(metaclass=Singleton):
    """Class to log results to files"""
    def __init__(self):
        cfg = Config()
        self._tags = ['ppo'] + cfg.tags
        implementation = get_autoschedular_impl()

        # Agent root = results_dir directly (v4_7_agent/, etc.)
        self.run_dir = str(get_agent_runs_root(cfg.results_dir, implementation))
        os.makedirs(self.run_dir, exist_ok=True)

        self._setup_dirs()

    def _setup_dirs(self):
        tags_file = os.path.join(self.run_dir, 'tags')
        with open(tags_file, 'w') as f:
            f.write('\n'.join(self._tags))
            f.write('\n')

        self.models_dir = os.path.join(self.run_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

        self.logs_dir = os.path.join(self.run_dir, 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)

        self.exec_data_file = os.path.join(self.logs_dir, 'exec_data.json')
        if not os.path.exists(self.exec_data_file):
            with open(self.exec_data_file, "w") as f:
                json.dump({}, f)

        self.train_dir = os.path.join(self.run_dir, 'train')
        os.makedirs(self.train_dir, exist_ok=True)
        self.train_results_file = os.path.join(self.train_dir, 'results.json')
        if not os.path.exists(self.train_results_file):
            with open(self.train_results_file, "w") as f:
                json.dump({}, f)

        self.eval_dir = os.path.join(self.run_dir, 'eval')
        os.makedirs(self.eval_dir, exist_ok=True)

        self.files_dict: dict[str, FileInstance] = {}

    def clear_per_iter_logs(self):
        """Clear per-iteration log files (entropy, reward, speedup, etc.).

        Call this on fresh training start to prevent accumulation across
        resume runs. Each FileInstance will recreate its file on next write.
        """
        per_iter_files = [
            'train/entropy', 'train/reward', 'train/final_speedup',
            'train_ppo/policy_loss', 'train_ppo/value_loss',
            'train_ppo/approx_kl', 'train_ppo/clip_frac',
            'train_ppo/clip_factor', 'train_ppo/entropy_loss',
            'eval/entropy', 'eval/reward', 'eval/cumulative_reward',
            'eval/average_speedup', 'eval/arithmetic_mean_speedup',
        ]
        for path in per_iter_files:
            full_path = os.path.join(self.logs_dir, path)
            if os.path.exists(full_path):
                os.remove(full_path)
        # Clear cached FileInstance objects so they recreate files
        self.files_dict.clear()

    def __getitem__(self, path: str):
        if path not in self.files_dict:
            full_path = os.path.join(self.logs_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
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
