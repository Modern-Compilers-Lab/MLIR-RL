import torch
from torch.utils.data import Dataset, DataLoader
from typing import Iterator
from rl_autoschedular import device
from rl_autoschedular.model import HiearchyModel as Model


T_timestep = tuple[
    int,            # num_loops
    torch.Tensor,   # action_index
    torch.Tensor,   # obs
    torch.Tensor,   # next_obs
    float,          # reward
    bool,           # done
]

# Only PPO-relevant attributes now
DYNAMIC_ATTRS = ['values', 'next_values', 'actions_old_log_p', 'returns', 'advantages']


class TrajectoryData(Dataset):
    """On-policy trajectory dataset for PPO."""

    num_loops: torch.Tensor
    actions_index: torch.Tensor
    obs: torch.Tensor
    next_obs: torch.Tensor
    rewards: torch.Tensor
    done: torch.Tensor

    values: torch.Tensor
    next_values: torch.Tensor
    actions_old_log_p: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    def __init__(
        self,
        num_loops: torch.Tensor,
        actions_index: torch.Tensor,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor
    ):
        self.num_loops = num_loops
        self.actions_index = actions_index
        self.obs = obs
        self.next_obs = next_obs
        self.rewards = rewards
        self.done = done

    def __len__(self) -> int:
        return self.obs.size(0)

    def __getitem__(self, idx: int):
        return (
            self.num_loops[idx],
            self.actions_index[idx],
            self.obs[idx],
            self.next_obs[idx],
            self.rewards[idx],
            self.done[idx],
            self.values[idx],
            self.next_values[idx],
            self.actions_old_log_p[idx],
            self.returns[idx],
            self.advantages[idx],
        )

    def __add__(self, other: 'TrajectoryData'):
        """Concatenate with another trajectory."""
        self_other = TrajectoryData(
            torch.cat((self.num_loops, other.num_loops)),
            torch.cat((self.actions_index, other.actions_index)),
            torch.cat((self.obs, other.obs)),
            torch.cat((self.next_obs, other.next_obs)),
            torch.cat((self.rewards, other.rewards)),
            torch.cat((self.done, other.done)),
        )
        for attr in DYNAMIC_ATTRS:
            if hasattr(self, attr) and hasattr(other, attr):
                self_val = getattr(self, attr)
                other_val = getattr(other, attr)
                assert isinstance(self_val, torch.Tensor) and isinstance(other_val, torch.Tensor)
                setattr(self_other, attr, torch.cat((self_val, other_val)))
        return self_other

    def loader(self, batch_size: int, shuffle: bool = True):
        """Create DataLoader for PPO training (uniform sampling)."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=False,
        )

    def copy(self) -> 'TrajectoryData':
        """Copy the trajectory."""
        self_copy = TrajectoryData(
            num_loops=self.num_loops.clone(),
            actions_index=self.actions_index.clone(),
            obs=self.obs.clone(),
            next_obs=self.next_obs.clone(),
            rewards=self.rewards.clone(),
            done=self.done.clone(),
        )
        for attr in DYNAMIC_ATTRS:
            if hasattr(self, attr):
                attr_val = getattr(self, attr)
                assert isinstance(attr_val, torch.Tensor)
                setattr(self_copy, attr, attr_val.clone())
        return self_copy

    def update_attributes(self, model: Model):
        """Update log-probs, values, returns, and advantages with the current model."""
        actions_old_log_p, values, _ = model(self.obs.to(device), self.actions_index)
        next_values = model.value_model(self.next_obs.to(device))

        self.actions_old_log_p = actions_old_log_p.cpu()
        self.values = values.cpu()
        self.next_values = next_values.cpu()

        self.__compute_returns()
        self.__compute_gae()

    def __compute_returns(self, gamma: float = 0.99):
        """Compute discounted returns (standard PPO style)."""
        self.returns = torch.zeros(len(self), dtype=torch.float32)
        last_return = 0.0
        for t in reversed(range(len(self))):
            mask = 1.0 - float(self.done[t])
            last_return = self.rewards[t] + gamma * last_return * mask
            self.returns[t] = last_return

    def __compute_gae(self, gamma: float = 0.99, lambda_: float = 0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        self.advantages = torch.zeros(len(self), dtype=torch.float32)
        last_advantage = 0.0
        for t in reversed(range(len(self))):
            mask = 1.0 - float(self.done[t])
            last_value = self.next_values[t] * mask
            last_advantage = last_advantage * mask

            delta = self.rewards[t] + gamma * last_value - self.values[t]
            last_advantage = delta + gamma * lambda_ * last_advantage

            self.advantages[t] = last_advantage


class TrajectoryCollector:
    """Collect timesteps into a trajectory for PPO."""

    def __init__(self):
        self.num_loops = []
        self.actions_index = []
        self.obs = []
        self.next_obs = []
        self.rewards = []
        self.done = []

    def append(self, timestep: T_timestep):
        self.num_loops.append(timestep[0])
        self.actions_index.append(timestep[1])
        self.obs.append(timestep[2])
        self.next_obs.append(timestep[3])
        self.rewards.append(timestep[4])
        self.done.append(timestep[5])

    def to_trajectory(self) -> TrajectoryData:
        return TrajectoryData(
            num_loops=torch.tensor(self.num_loops, dtype=torch.int64),
            actions_index=torch.cat(self.actions_index),
            obs=torch.cat(self.obs),
            next_obs=torch.cat(self.next_obs),
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            done=torch.tensor(self.done, dtype=torch.bool),
        )

    def reset(self):
        self.num_loops.clear()
        self.actions_index.clear()
        self.obs.clear()
        self.next_obs.clear()
        self.rewards.clear()
        self.done.clear()
