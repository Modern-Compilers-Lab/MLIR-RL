import torch
from torch.utils.data import Dataset, DataLoader, Sampler, RandomSampler
from typing import Iterator, Optional
from rl_autoschedular import device
from rl_autoschedular.model import HiearchyModel as Model
from time import time
from utils.config import Config
from utils.log import print_info


T_timestep = tuple[
    int,  # num_loops
    torch.Tensor,  # action_index
    torch.Tensor,  # obs
    torch.Tensor,  # next_obs
    float,  # action_bev_log_p
    float,  # reward
    bool,  # done
]

DYNAMIC_ATTRS = ['values', 'next_values', 'actions_old_log_p', 'off_policy_rates', 'returns', 'advantages']


class TopKAdvantageSampler(Sampler[int]):
    """
    A Sampler that yields a random permutation of the indices
    corresponding to the top-K highest advantage values in a trajectory.

    Args:
        data_source (TrajectoryData): The dataset, which must have a `get_all_advantages` method.
        num_samples (int): The maximum number of samples to take batches from.
    """
    def __init__(self, data_source: 'TrajectoryData', num_samples: int):
        self.data_source = data_source
        self.num_samples = num_samples

        # Get all advantage values from the dataset
        advantages = self.data_source.advantages

        # Ensure we don't request more samples than available
        self.num_samples = min(self.num_samples, advantages.size(0))

        _, self.top_k_indices = torch.topk(advantages.abs(), k=self.num_samples)

    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over shuffled indices of the top-k samples.
        This is called by the DataLoader at the start of each epoch.
        """
        # Shuffle the top-k indices to ensure random order
        shuffled_indices = self.top_k_indices[torch.randperm(self.num_samples)]

        # Yield the indices one by one
        yield from shuffled_indices.tolist()

    def __len__(self) -> int:
        """The total number of samples to be drawn."""
        return self.num_samples


class TrajectoryData(Dataset):
    """Dataset to store the trajectory data.

    Args:
            num_loops (torch.Tensor): Number of loops in the trajectory.
            actions_index (torch.Tensor): Actions indices in the trajectory.
            obs (torch.Tensor): Observations in the trajectory.
            next_obs (torch.Tensor): Observations of next states in the trajectory.
            actions_bev_log_p (torch.Tensor): Action log probabilities following behavioral policy.
            rewards (torch.Tensor): Rewards in the trajectory.
            done (torch.Tensor): Done flags in the trajectory.
    """
    sizes: list[int]
    """Sizes of all the included trajectories"""

    num_loops: torch.Tensor
    """Number of loops in the trajectory."""
    actions_index: torch.Tensor
    """Actions in the trajectory."""
    obs: torch.Tensor
    """Observations in the trajectory"""
    next_obs: torch.Tensor
    """Observations of next states in the trajectory."""
    actions_bev_log_p: torch.Tensor
    """Action log probabilities following behavioral policy in the trajectory."""
    rewards: torch.Tensor
    """Rewards in the trajectory."""
    done: torch.Tensor
    """Done flags in the trajectory."""

    values: torch.Tensor
    """Values of actions in the trajectory."""
    next_values: torch.Tensor
    """Values of actions in the trajectory with one additional step (shifted to one step in the future)."""
    actions_old_log_p: torch.Tensor
    """Action log probabilities following old policy in the trajectory."""
    off_policy_rates: torch.Tensor
    """Off-policy rates (rho) for the current policy."""
    returns: torch.Tensor
    """Returns in the trajectory."""
    advantages: torch.Tensor
    """Advantages in the trajectory."""

    def __init__(
        self,
        num_loops: torch.Tensor,
        actions_index: torch.Tensor,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        actions_bev_log_p: torch.Tensor,
        rewards: torch.Tensor,
        done: torch.Tensor
    ):
        self.num_loops = num_loops
        self.actions_index = actions_index
        self.obs = obs
        self.next_obs = next_obs
        self.actions_bev_log_p = actions_bev_log_p
        self.rewards = rewards
        self.done = done

        self.sizes = [len(self)]

    def __len__(self) -> int:
        """Get the length of the trajectory.

        Returns:
            int: The length of the trajectory.
        """
        return self.obs.size(0)

    def __getitem__(self, idx: int):
        """Get a single timestep from the trajectory.

        Args:
            idx (int): Index of the timestep to retrieve.

        Returns:
            tuple: A tuple containing the timestep data.
        """
        return (
            self.num_loops[idx],
            self.actions_index[idx],
            self.obs[idx],
            self.next_obs[idx],
            self.actions_bev_log_p[idx],
            self.rewards[idx],
            self.done[idx],

            self.values[idx],
            self.next_values[idx],
            self.actions_old_log_p[idx],
            self.off_policy_rates[idx],
            self.returns[idx],
            self.advantages[idx],
        )

    def __add__(self, other: 'TrajectoryData'):
        """Concatenate this trajectory with another.

        Args:
            other (TrajectoryData): The other trajectory to concatenate with

        Returns:
            TrajectoryData: The trajectory containing both
        """
        self_other_sizes = self.sizes + other.sizes

        # Truncate to 10 trajectories
        self_other_sizes = self_other_sizes[-Config().replay_count:]
        start = - sum(self_other_sizes)
        assert len(self_other_sizes) <= Config().replay_count

        self_other = TrajectoryData(
            torch.cat((self.num_loops, other.num_loops))[start:],
            torch.cat((self.actions_index, other.actions_index))[start:],
            torch.cat((self.obs, other.obs))[start:],
            torch.cat((self.next_obs, other.next_obs))[start:],
            torch.cat((self.actions_bev_log_p, other.actions_bev_log_p))[start:],
            torch.cat((self.rewards, other.rewards))[start:],
            torch.cat((self.done, other.done))[start:],
        )
        for attr in DYNAMIC_ATTRS:
            if hasattr(self, attr) and hasattr(other, attr):
                self_val = getattr(self, attr)
                other_val = getattr(other, attr)
                assert isinstance(self_val, torch.Tensor) and isinstance(other_val, torch.Tensor)
                setattr(self_other, attr, torch.cat(self_val, other_val)[start:])

        self_other.sizes = self_other_sizes

        assert len(self_other) == sum(self_other_sizes)

        return self_other

    def loader(self, batch_size: Optional[int], num_trajectories: int):
        """Create a DataLoader for the trajectory.

        Args:
            batch_size (int): Batch size for the DataLoader.
            num_samples (int): Maximum number of samples to take batches from.

        Returns:
            DataLoader: The DataLoader for the trajectory.
        """
        num_samples = sum(self.sizes[-num_trajectories:])
        if batch_size is None:
            batch_size = num_samples
        match Config().reuse_experience:
            case 'topk':
                sampler = TopKAdvantageSampler(self, num_samples)
            case 'random':
                sampler = RandomSampler(self, num_samples=num_samples)
            case 'none':
                sampler = None

        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            pin_memory=device.type != 'cpu',
        )

    def copy(self) -> 'TrajectoryData':
        """Copy the trajectory.

        Returns:
            TrajectoryData: The copied trajectory.
        """
        self_copy = TrajectoryData(
            num_loops=self.num_loops.clone(),
            actions_index=self.actions_index.clone(),
            obs=self.obs.clone(),
            next_obs=self.next_obs.clone(),
            actions_bev_log_p=self.actions_bev_log_p.clone(),
            rewards=self.rewards.clone(),
            done=self.done.clone(),
        )
        for attr in DYNAMIC_ATTRS:
            if hasattr(self, attr):
                attr_val = getattr(self, attr)
                assert isinstance(attr_val, torch.Tensor)
                setattr(self_copy, attr, attr_val.clone())

        self_copy.sizes = self.sizes.copy()

        return self_copy

    def update_attributes(self, model: Model):
        """Update the attributes of the trajectory following the new model.

        Args:
            model (Model): The model to use for updating the attributes.
        """
        start = time()

        actions_old_log_p, values, _ = model(self.obs.to(device), self.actions_index.to(device))
        self.actions_old_log_p, self.values = actions_old_log_p.cpu(), values.cpu()

        self.next_values = model.value_model(self.next_obs.to(device)).cpu()

        self.__compute_rho()
        self.__compute_returns()
        self.__compute_gae()
        end = time()
        time_ms = int((end - start) * 1000)
        print_info(f"Updated {len(self)} attributes in {time_ms}ms")

    def __compute_rho(self) -> torch.Tensor:
        """Compute the off-policy rate (rho) for the current policy.

        Returns:
            torch.Tensor: The off-policy rate.
        """
        if 'epsilon' not in Config().exploration and Config().reuse_experience == 'none':
            self.off_policy_rates = torch.ones_like(self.actions_bev_log_p)
            return

        self.off_policy_rates = torch.exp(torch.clamp(self.actions_old_log_p - self.actions_bev_log_p, -80.0, 80.0))

    def __compute_returns(self, gamma: float = 1.0) -> torch.Tensor:
        """Compute the returns.

        Args:
            done (torch.Tensor): done flags.
            rewards (torch.Tensor): rewards.
            gamma (float): discount factor. Defaults to 1.

        Returns:
            torch.Tensor: returns.
        """
        self.returns = torch.zeros(len(self), dtype=torch.float32)
        last_return = 0

        for t in reversed(range(len(self))):
            mask = ~self.done[t]
            last_return = last_return * mask

            last_return = self.values[t] + (self.rewards[t] + gamma * last_return - self.values[t]) * self.off_policy_rates[t].clamp_max(1)

            self.returns[t] = last_return

    def __compute_gae(self, gamma: float = 1.0, lambda_: float = 0.95) -> torch.Tensor:
        """Compute the Generalized Advantage Estimation.

        Args:
            gamma (float): discount factor.
            lambda_ (float): GAE factor.

        Returns:
            torch.Tensor: advantages.
            torch.Tensor: returns.
        """
        self.advantages = torch.zeros(len(self), dtype=torch.float32)
        last_advantage = 0

        for t in reversed(range(len(self))):
            mask = ~self.done[t]
            last_value = self.next_values[t] * mask
            last_advantage = last_advantage * mask

            delta = self.rewards[t] + gamma * last_value - self.values[t]
            last_advantage = delta + gamma * lambda_ * last_advantage

            self.advantages[t] = last_advantage


class TrajectoryCollector:
    """Class that appends timestep data to a trajectory."""

    num_loops: list[int]
    """Number of loops in the trajectory."""
    actions_index: list[torch.Tensor]
    """Actions in the trajectory."""
    obs: list[torch.Tensor]
    """Observations in the trajectory."""
    next_obs: list[torch.Tensor]
    """Observations of next states in the trajectory."""
    actions_bev_log_p: list[float]
    """Action log probabilities following behavioral policy in the trajectory."""
    rewards: list[float]
    """Rewards in the trajectory."""
    done: list[bool]
    """Done flags in the trajectory."""

    def __init__(self):
        """Initialize the trajectory collector."""
        self.num_loops = []
        self.actions_index = []
        self.obs = []
        self.next_obs = []
        self.actions_bev_log_p = []
        self.rewards = []
        self.done = []

    def __add__(self, other: 'TrajectoryCollector'):
        self.num_loops.extend(other.num_loops)
        self.actions_index.extend(other.actions_index)
        self.obs.extend(other.obs)
        self.next_obs.extend(other.next_obs)
        self.actions_bev_log_p.extend(other.actions_bev_log_p)
        self.rewards.extend(other.rewards)
        self.done.extend(other.done)

        return self

    def append(self, timestep: T_timestep):
        """Append a single timestep to the trajectory.

        Args:
            timestep (T_timestep): The timestep data to append.
        """
        self.num_loops.append(timestep[0])
        self.actions_index.append(timestep[1])
        self.obs.append(timestep[2])
        self.next_obs.append(timestep[3])
        self.actions_bev_log_p.append(timestep[4])
        self.rewards.append(timestep[5])
        self.done.append(timestep[6])

    def to_trajectory(self) -> TrajectoryData:
        """Convert the collected data to a TrajectoryData object.

        Returns:
            TrajectoryData: The trajectory containing all collected data.
        """
        return TrajectoryData(
            num_loops=torch.tensor(self.num_loops, dtype=torch.int64),
            actions_index=torch.cat(self.actions_index),
            obs=torch.cat(self.obs),
            next_obs=torch.cat(self.next_obs),
            actions_bev_log_p=torch.tensor(self.actions_bev_log_p, dtype=torch.float32),
            rewards=torch.tensor(self.rewards, dtype=torch.float32),
            done=torch.tensor(self.done, dtype=torch.bool),
        )

    def reset(self):
        """Reset the trajectory collector."""
        self.num_loops.clear()
        self.actions_index.clear()
        self.obs.clear()
        self.next_obs.clear()
        self.actions_bev_log_p.clear()
        self.rewards.clear()
        self.done.clear()
