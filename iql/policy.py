import torch
import torch.nn as nn
from torch.distributions import Distribution
from typing import Optional, List, Type
from rl_autoschedular import config as cfg
from rl_autoschedular.actions import ActionSpace, Interchange
from rl_autoschedular.observation import Observation, ObservationPart

# Match PPO’s activation config
ACTIVATION = nn.ReLU if cfg.activation == "relu" else nn.Tanh


class IQLPolicyModel(nn.Module):
    """
    IQL policy network, sharing architecture with PPO’s PolicyModel.
    - Backbone: 3×512 MLP with ACTIVATION()
    - Heads: one for action selection + one per action’s parameterization
    - Output: list[Distribution], via ActionSpace.distributions
    - Loss: BC loss with advantage-weighted log-likelihood (AWAC / IQL style)
    """

    def __init__(self, obs_parts: List[Type[ObservationPart]]):
        super().__init__()
        self.obs_parts = obs_parts


        # Shared encoder
        in_size = sum(part.size() for part in obs_parts)
        self.backbone = nn.Sequential(
            nn.Linear(in_size, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
        )

        # One head for action choice + one for each action’s params
        output_sizes = [ActionSpace.size()] + [
            action.network_output_size() for action in ActionSpace.supported_actions
        ]
        self.heads_attributes = [f"head_{i}" for i in range(len(output_sizes))]

        for head_attr, output_size in zip(self.heads_attributes, output_sizes):
            if not output_size:
                setattr(self, head_attr, None)
                continue

            head = nn.Linear(512, output_size)
            if cfg.new_architecture:
                head = nn.Sequential(nn.Linear(512, 512), ACTIVATION(), head)
            setattr(self, head_attr, head)

    def forward(self, obs: torch.Tensor) -> List[Optional[Distribution]]:
        """
        Forward pass: produce a Distribution object per action head.
        """
        embedded = self.backbone(Observation.get_parts(obs, *self.obs_parts))
        heads: List[Optional[nn.Module]] = [getattr(self, attr) for attr in self.heads_attributes]
        actions_logits = [head(embedded) if head else None for head in heads]
        return ActionSpace.distributions(obs, *actions_logits)

    def loss(
        self,
        actions_log_p: torch.Tensor,
        advantages: torch.Tensor,
        beta: float = 1.0,
    ) -> torch.Tensor:
        """
        Advantage-weighted behavioral cloning (AWAC) / IQL policy loss.
        Args:
            obs: Observations [B, ...]
            actions_log_p: log π(a|s) from this policy, evaluated at dataset actions
            advantages: Advantage estimates A(s,a) from IQL (Q - V)
            beta: Temperature scaling (larger beta = more deterministic)
        Returns:
            Scalar loss tensor.
        """
        # Weights = exp(A / beta), clipped for stability
        weights = torch.exp(advantages / beta).clamp(max=100.0)
        loss = -(weights * actions_log_p).mean()
        return loss