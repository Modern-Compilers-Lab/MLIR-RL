import os
import numpy as np
import torch
from utils.singleton import Singleton
from utils.log import print_success

class OfflineDataset(metaclass=Singleton):
    """Singleton class to collect and store trajectories for offline RL """

    def __init__(self, save_dir: str = "offline_data", fname: str = "dataset.npz"):
        """
        Args:
            save_dir (str): Directory to store dataset.
            fname (str): Dataset filename.
        """
        self.save_dir = save_dir
        self.fname = fname
        os.makedirs(self.save_dir, exist_ok=True)

        self.buffer = []  # in-memory buffer for efficiency
        self.file_path = os.path.join(self.save_dir, self.fname)

    def add_transition(self, obs, action, next_obs, reward, done):
        """Add one transition to buffer."""
        self.buffer.append({
            "obs": obs.squeeze(0).cpu().numpy() if torch.is_tensor(obs) else obs,
            "action": action.detach().cpu().numpy().squeeze(0) if torch.is_tensor(action) else np.array(action).squeeze(0),
            "next_obs": next_obs.squeeze(0).cpu().numpy() if torch.is_tensor(next_obs) else next_obs,
            "reward": float(reward),
            "done": bool(done),
        })

    def add_trajectory(self, trajectory):
        """Add a full trajectory (list of transitions)."""
        self.buffer.extend(trajectory)

    def flush(self):
        """Save buffer to disk as npz and clear it."""
        if not self.buffer:
            return

        # Convert buffer to arrays
        obs = np.array([t["obs"] for t in self.buffer], dtype=np.float32)
        actions = np.array([t["action"] for t in self.buffer], dtype=np.int64)
        next_obs = np.array([t["next_obs"] for t in self.buffer], dtype=np.float32)
        rewards = np.array([t["reward"] for t in self.buffer], dtype=np.float32)
        dones = np.array([t["done"] for t in self.buffer], dtype=np.bool_)

        if os.path.exists(self.file_path):
            # If file exists, append to it
            old = np.load(self.file_path)
            obs = np.concatenate([old["obs"], obs], axis=0)
            actions = np.concatenate([old["actions"], actions], axis=0)
            next_obs = np.concatenate([old["next_obs"], next_obs], axis=0)
            rewards = np.concatenate([old["rewards"], rewards], axis=0)
            dones = np.concatenate([old["dones"], dones], axis=0)

        np.savez_compressed(
            self.file_path,
            obs=obs,
            actions=actions,
            next_obs=next_obs,
            rewards=rewards,
            dones=dones,
        )

        print_success(f"[OfflineDataset] Flushed {len(self.buffer)} transitions -> {self.file_path}")
        self.buffer.clear()

    def load(self, mmap_mode=None):
        """Load dataset from disk as dict of numpy arrays (optionally memory-mapped)."""
        if not os.path.exists(self.file_path):
            return {}
        return np.load(self.file_path, mmap_mode=mmap_mode)