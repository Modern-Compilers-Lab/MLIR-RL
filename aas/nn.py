from aas import config as cfg
from aas.state import OperationState
import torch
import torch.nn as nn
from typing import Literal, Optional


class AASNetwork(torch.nn.Module):
    """Class to represent the AAS network."""

    input_dim: int
    """The input dimension of the AlphaAutoScheduler network model. It matches the feature tensor dimension of the operation state."""
    select_network_dim: int
    """The output dimension of the select model. The output tensor is a probability distribution over transformations."""
    select_network: nn.Sequential
    """The select network. It outputs a probability distribution over transformations."""
    parallel_params_network_dim: int
    """The output dimension of the parallelization parameters model. The output tensor is a probability distribution over tiling sizes."""
    parallel_params_networks: list[nn.Sequential]
    """The parallelization parameters networks. Each network is responsible for a specific loop in the operation."""
    value_output_dim: int
    """The output dimension of the value model. The output tensor is a single value representing the value of the operation"""
    value_network: nn.Sequential
    """The value network. It outputs a single value representing the value of the operation."""

    def __init__(self):
        """Initialize the AAS network."""
        super(AASNetwork, self).__init__()
        # Initialize the input dimension
        self.input_dim = OperationState.get_tensor_dim()
        # Define the backbone of the policy network
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        # Define the output layers of the policy network
        self.select_network_dim = cfg.num_transformations
        self.select_network = nn.Linear(512, self.select_network_dim)
        self.parallel_params_network_dim = cfg.max_num_loops * (cfg.num_tile_sizes + 1)
        self.parallel_params_networks = [nn.Linear(512, cfg.num_tile_sizes + 1) for _ in range(cfg.max_num_loops)]
        # Define the output layers of the value network
        self.value_output_dim = 1
        self.value_network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, obs: torch.Tensor):
        """Forward pass of the AAS network."""
        x = self.backbone(obs)
        select_logits = self.select_network(x)
        parallel_params_logits = torch.concatenate([parallel_params_network(x).unsqueeze(1) for parallel_params_network in self.parallel_params_networks], dim=1)
        value = self.value_network(obs)
        return select_logits, parallel_params_logits, value.squeeze(-1)

    def eval_policy(self, obs: torch.Tensor):
        """Evaluate the policy network."""
        x = self.backbone(obs)
        select_logits = self.select_network(x)
        parallel_params_logits = torch.concatenate([parallel_params_network(x).unsqueeze(1) for parallel_params_network in self.parallel_params_networks], dim=1)
        return select_logits, parallel_params_logits

    def eval_value(self, obs: torch.Tensor):
        """Evaluate the value network."""
        return self.value_network(obs).squeeze(-1)


class CrossEntropyLoss:
    """Class to represent the cross-entropy loss function."""

    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        """Initialize the cross-entropy loss function."""
        self.reduction = reduction

    def __call__(self, y_pred: torch.Tensor, y_target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Compute the cross-entropy loss.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_target (torch.Tensor): The target values.
            mask (Optional[torch.Tensor], optional): The mask to apply to the loss. Defaults to None.

        Returns:
            torch.Tensor: The loss value.
        """
        loss = -torch.sum(y_target * torch.log(y_pred), dim=1)
        if mask is not None:
            loss = loss * mask
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction type {self.reduction}")


class BSELoss:
    """Class to represent the Boltzmann Squared error loss function."""

    def __init__(self, reduction: Literal['mean', 'sum'] = 'mean'):
        """Initialize the Boltzmann Squared error loss function."""
        self.reduction = reduction

    def __call__(self, y_pred: torch.Tensor, y_target: torch.Tensor):
        """Compute the Boltzmann Squared error loss.

        Args:
            y_pred (torch.Tensor): The predicted values.
            y_target (torch.Tensor): The target values.

        Returns:
            torch.Tensor: The loss value.
        """
        a = cfg.bse_param
        r = cfg.bse_relaxation
        loss = r * ((torch.exp(a * y_target) / 2) * (y_target - y_pred)) ** 2
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise ValueError(f"Invalid reduction type {self.reduction}")
