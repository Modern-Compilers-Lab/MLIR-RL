import torch
import torch.nn as nn
from torch.distributions import Distribution
from typing import Optional
from rl_autoschedular.actions import ActionSpace, Interchange
from rl_autoschedular.observation import OpFeatures, ActionHistory, ProducerOpFeatures, Observation
from utils.config import Config


ACTIVATION = nn.ReLU


class HiearchyModel(nn.Module):
    """Hierarchical reinforcement learning model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(HiearchyModel, self).__init__()

        self.policy_model = PolicyModel()
        self.value_model = ValueModel()

    def __call__(self, obs: torch.Tensor, actions_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(obs, actions_index)

    def forward(self, obs: torch.Tensor, actions_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            obs (torch.Tensor): The input tensor.
            actions_index (torch.Tensor): The list of actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        actions_log_p, entropies = ActionSpace.distributions_stats(self.policy_model(obs), actions_index)

        values = self.value_model(obs)

        return actions_log_p, values, entropies

    def sample(self, obs: torch.Tensor, greedy: bool = False, eps: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the model.

        Args:
            obs (torch.Tensor): The input tensor.
            greedy (bool): Whether to sample greedily.
            eps (Optional[float]): Epsilon value for exploration. Defaults to None.

        Returns:
            torch.Tensor: Sampled actions index.
            torch.Tensor: actions log probability.
            torch.Tensor: resulting entropy.
        """
        assert not greedy or eps is None, 'Cannot be greedy and explore at the same time.'

        # Model feedforward
        distributions = self.policy_model(obs)
        eps_distributions = ActionSpace.uniform_distributions(obs)
        actions_index = ActionSpace.sample(
            obs,
            distributions,
            eps_distributions,
            uniform=eps is not None and torch.rand(1).item() < eps,
            greedy=greedy
        )
        actions_log_p, entropies = ActionSpace.distributions_stats(
            distributions,
            actions_index,
            eps_distributions=eps_distributions if eps is not None else None,
            eps=eps
        )

        return actions_index, actions_log_p, entropies


class ValueModel(nn.Module):
    """Value model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(ValueModel, self).__init__()

        self.lstm = LSTMEmbedding()

        self.network = nn.Sequential(
            nn.Linear(self.lstm.output_size, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
            nn.Linear(512, 1),
        )

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return super().__call__(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            obs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The value tensor.
        """
        return self.network(self.lstm(obs)).squeeze(-1)

    def loss(self, new_values: torch.Tensor, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Calculate the value loss.

        Args:
            new_values (torch.Tensor): The new value tensor.
            values (torch.Tensor): The value tensor.
            returns (torch.Tensor): The returns tensor.

        Returns:
            torch.Tensor: The value loss.
        """
        if Config().value_clip:
            vclip = values + torch.clamp(new_values - values, -0.2, 0.2)
            vloss1 = (returns - vclip).pow(2)
            vloss2 = (returns - new_values).pow(2)
            return torch.max(vloss1, vloss2).mean()
        return (returns - new_values).pow(2).mean()


class PolicyModel(nn.Module):
    """Policy model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(PolicyModel, self).__init__()

        self.log_std = Interchange.log_std

        self.lstm = LSTMEmbedding()

        self.backbone = nn.Sequential(
            nn.Linear(self.lstm.output_size, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
            nn.Linear(512, 512),
            ACTIVATION(),
        )

        output_sizes = [ActionSpace.size()] + [action.network_output_size() for action in ActionSpace.supported_actions]
        self.heads = nn.ModuleList()
        for output_size in output_sizes:
            if not output_size:
                self.heads.append(None)
                continue
            self.heads.append(nn.Sequential(
                nn.Linear(512, 512),
                ACTIVATION(),
                nn.Linear(512, output_size)
            ))

    def __call__(self, obs: torch.Tensor) -> list[Optional[Distribution]]:
        return super().__call__(obs)

    def forward(self, obs: torch.Tensor) -> list[Optional[Distribution]]:
        """Forward pass of the model.

        Args:
            obs (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        embedded = self.backbone(self.lstm(obs))
        actions_logits = [head(embedded) if head else None for head in self.heads]

        return ActionSpace.distributions(obs, *actions_logits)

    def loss(self, actions_log_p: torch.Tensor, actions_bev_log_p: torch.Tensor, off_policy_rates: torch.Tensor, advantages: torch.Tensor, clip_range: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the policy loss.

        Args:
            new_actions_log_p (torch.Tensor): The log probabilities of the new actions.
            actions_bev_log_p (torch.Tensor): The log probabilities of the actions under the behavior policy.
            off_policy_rates (torch.Tensor): The rate between the old policy and the behavioral (mu) policy.
            advantages (torch.Tensor): The advantages of the actions.
            clip_range (float): The clipping range for the policy loss.

        Returns:
            torch.Tensor: The policy loss.
            float: The ratio clip fraction (for logging purposes)
        """
        ratios = torch.exp(torch.clamp(actions_log_p - actions_bev_log_p, -80.0, 80.0))
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, (1 - clip_range) * off_policy_rates, (1 + clip_range) * off_policy_rates) * advantages
        clip_frac = (torch.abs((ratios / off_policy_rates - 1)) > clip_range).float().mean()
        return - torch.min(surr1, surr2).mean(), clip_frac


class LSTMEmbedding(nn.Module):
    def __init__(self):
        super(LSTMEmbedding, self).__init__()

        embedding_size = 411

        self.output_size = embedding_size + ActionHistory.size()

        self.embedding = nn.Sequential(
            nn.Linear(OpFeatures.size(), 512),
            nn.ELU(),
            nn.Dropout(0.225),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Dropout(0.225),
        )

        self.lstm = nn.LSTM(512, embedding_size)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return super().__call__(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        consumer_feats = Observation.get_part(obs, OpFeatures)
        producer_feats = Observation.get_part(obs, ProducerOpFeatures)

        consumer_embeddings = self.embedding(consumer_feats).unsqueeze(0)
        producer_embeddings = self.embedding(producer_feats).unsqueeze(0)

        _, (final_hidden, _) = self.lstm(torch.cat((consumer_embeddings, producer_embeddings)))

        return torch.cat((final_hidden.squeeze(0), Observation.get_part(obs, ActionHistory)), 1)
