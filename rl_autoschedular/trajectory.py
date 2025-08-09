import torch
from torch.utils.data import Dataset, DataLoader
from rl_autoschedular import config as cfg
from rl_autoschedular.model import HiearchyModel as Model


T_timestep = tuple[
    int,  # num_loops
    torch.Tensor,  # action_index
    torch.Tensor,  # obs
    torch.Tensor,  # next_obs
    float,  # action_bev_log_p
    float,  # reward
    bool,  # done
]

T_data_timestep = tuple[
    *T_timestep,

    float,  # value
    float,  # next_value
    float,  # action_old_log_p
    float,  # off_policy_rate
    float,  # return
    float,  # advantage
]

DYNAMIC_ATTRS = ['values', 'next_values', 'actions_old_log_p', 'off_policy_rates', 'returns', 'advantages']


class TrajectoryData(Dataset[T_data_timestep]):
    """Dataset to store the trajectory data."""

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
        """Initialize the trajectory data.

        Args:
            num_loops (torch.Tensor): Number of loops in the trajectory.
            actions_index (torch.Tensor): Actions indices in the trajectory.
            obs (torch.Tensor): Observations in the trajectory.
            next_obs (torch.Tensor): Observations of next states in the trajectory.
            actions_bev_log_p (torch.Tensor): Action log probabilities following behavioral policy.
            rewards (torch.Tensor): Rewards in the trajectory.
            done (torch.Tensor): Done flags in the trajectory.
        """
        self.num_loops = num_loops
        self.actions_index = actions_index
        self.obs = obs
        self.next_obs = next_obs
        self.actions_bev_log_p = actions_bev_log_p
        self.rewards = rewards
        self.done = done

    def __len__(self) -> int:
        """Get the length of the trajectory.

        Returns:
            int: The length of the trajectory.
        """
        return self.obs.size(0)

    def __getitem__(self, idx: int) -> T_data_timestep:
        """Get a single timestep from the trajectory.

        Args:
            idx (int): Index of the timestep to retrieve.

        Returns:
            tuple: A tuple containing the timestep data.
        """
        return (
            self.num_loops[idx].item(),
            self.actions_index[idx],
            self.obs[idx],
            self.next_obs[idx],
            self.actions_bev_log_p[idx].item(),
            self.rewards[idx].item(),
            self.done[idx].item(),

            self.values[idx].item(),
            self.next_values[idx].item(),
            self.actions_old_log_p[idx].item(),
            self.off_policy_rates[idx].item(),
            self.returns[idx].item(),
            self.advantages[idx].item(),
        )

    def __add__(self, other: 'TrajectoryData'):
        """Concatenate this trajectory with another.

        Args:
            other (TrajectoryData): The other trajectory to concatenate with

        Returns:
            TrajectoryData: The trajectory containing both
        """
        self_other = TrajectoryData(
            torch.cat((self.num_loops, other.num_loops)),
            torch.cat((self.actions_index, other.actions_index)),
            torch.cat((self.obs, other.obs)),
            torch.cat((self.next_obs, other.next_obs)),
            torch.cat((self.actions_bev_log_p, other.actions_bev_log_p)),
            torch.cat((self.rewards, other.rewards)),
            torch.cat((self.done, other.done)),
        )
        for attr in DYNAMIC_ATTRS:
            if hasattr(self, attr) and hasattr(other, attr):
                self_val = getattr(self, attr)
                other_val = getattr(other, attr)
                assert isinstance(self_val, torch.Tensor) and isinstance(other_val, torch.Tensor)
                setattr(self_other, attr, torch.cat(self_val, other_val))

        return self_other

    def loader(self, batch_size: int, shuffle: bool = True):
        """Create a DataLoader for the trajectory.

        Args:
            batch_size (int, optional): Batch size for the DataLoader. Defaults to cfg.ppo_batch_size.
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True.

        Returns:
            DataLoader: The DataLoader for the trajectory.
        """
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

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

        return self_copy

    def update_attributes(self, model: Model):
        """Update the attributes of the trajectory following the new model.

        Args:
            model (Model): The model to use for updating the attributes.
        """
        self.actions_old_log_p, self.values, _ = model(self.obs, self.actions_index)
        self.next_values = model.value_model(self.next_obs)

        self.__compute_rho()
        self.__compute_returns()
        self.__compute_gae()

    def __compute_rho(self) -> torch.Tensor:
        """Compute the off-policy rate (rho) for the current policy.

        Returns:
            torch.Tensor: The off-policy rate.
        """
        self.off_policy_rates = torch.exp(torch.clamp(self.actions_old_log_p - self.actions_bev_log_p, -80.0, 80.0))
        if 'epsilon' not in cfg.exploration and not cfg.reuse_experience:
            assert (self.off_policy_rates == 1).all(), 'off_policy_rates should be 1 since behavior policy is the same as the current policy.'

    def __compute_returns(self, gamma: float = 0.99) -> torch.Tensor:
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

            last_return = self.values[t] + (self.rewards[t] + gamma * last_return - self.values[t]) * self.off_policy_rates[t]

            self.returns[t] = last_return

    def __compute_gae(self, gamma: float = 0.99, lambda_: float = 0.95) -> torch.Tensor:
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
            done=torch.tensor(self.done, dtype=torch.bool)
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
