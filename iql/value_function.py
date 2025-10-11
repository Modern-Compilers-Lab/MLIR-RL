import torch
import torch.nn as nn
from typing import List, Type
from rl_autoschedular import config as cfg
from rl_autoschedular.observation import Observation, ObservationPart


ACTIVATION = nn.ReLU


class IQLValueModel(nn.Module):
    """
    IQL Value function with the SAME encoder/MLP layout as PPO's ValueModel:
      Linear(sum(obs_parts)->512) -> ACT -> 512 -> ACT -> 512 -> ACT -> 1

    - Input: full Observation tensor, then sliced via Observation.get_parts to match PPO.
    - Output: V(s) as shape [B], same squeeze(-1) behavior as PPO.
    - Loss: Expectile regression with parameter tau (IQL).
    """

    def __init__(
        self,
        obs_parts: List[Type[ObservationPart]],
        tau: float = 0.7,
    ):
        super().__init__()
        self.obs_parts = obs_parts
        self.tau = cfg.tau  # consider wiring this from cfg (e.g., cfg.iql.tau) if you keep hyperparams in config

        in_size = sum(part.size() for part in obs_parts)
        self.network = nn.Sequential(
            nn.Linear(in_size, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
            nn.Linear(512, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: full Observation tensor (like in PPO)
        Returns:
            V(s) as [B]
        """
        x = Observation.get_parts(obs, *self.obs_parts)
        return self.network(x).squeeze(-1)  # [B]

    @torch.no_grad()
    def v(self, obs: torch.Tensor) -> torch.Tensor:
        """Convenience alias often used in IQL codepaths."""
        return self.forward(obs)

    def loss(self, obs: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
        """
        Expectile regression loss: minimize E[ w_tau(u) * u^2 ], u = Q(s,a) - V(s)

        Args:
            obs: full Observation tensor for states [B, ...] (same as PPO input)
            q_values: [B] or [B,1] tensor with target Q(s,a) (DETACHED upstream in IQL)
        """
        v = self.forward(obs)                     # [B]
        q = q_values.squeeze(-1)                  # [B]
        diff = q - v                              # u

        # weight = |tau - 1(u < 0)|
        # same as: tau if u >= 0 else (1 - tau)
        weight = torch.abs(self.tau - (diff < 0).float())
        return (weight * diff.pow(2)).mean()
