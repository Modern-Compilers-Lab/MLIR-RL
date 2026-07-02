import torch
import torch.nn as nn
from torch.distributions import Distribution
from typing import Optional
from rl_autoschedular_v4_9.actions import ActionSpace, Interchange
from rl_autoschedular_v4_9.observation import OpFeatures, ActionHistory, ProducerOpFeatures, HardwareFeatures, Observation, NumLoops
from rl_autoschedular_v4_9.state import OperationType
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

        self.embedding = TransformerEmbedding()

        self.network = nn.Sequential(
            nn.Linear(self.embedding.output_size, 512),
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
        return self.network(self.embedding(obs)).squeeze(-1)

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

        self.embedding = TransformerEmbedding()

        self.backbone = nn.Sequential(
            nn.Linear(self.embedding.output_size, 512),
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
        embedded = self.backbone(self.embedding(obs))
        actions_logits = [head(embedded) if head else None for head in self.heads]

        return ActionSpace.distributions(obs, *actions_logits)

    def loss(self, actions_log_p: torch.Tensor, actions_bev_log_p: torch.Tensor, off_policy_rates: torch.Tensor, advantages: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the policy loss.

        Args:
            new_actions_log_p (torch.Tensor): The log probabilities of the new actions.
            actions_bev_log_p (torch.Tensor): The log probabilities of the actions under the behavior policy.
            off_policy_rates (torch.Tensor): The rate between the old policy and the behavioral (mu) policy.
            advantages (torch.Tensor): The advantages of the actions.

        Returns:
            torch.Tensor: The policy loss.
            float: The ratio clip fraction (for logging purposes)
        """
        clip_range = Config().ppo_clip_range
        ratios = torch.exp(torch.clamp(actions_log_p - actions_bev_log_p, -80.0, 80.0))
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, (1 - clip_range) * off_policy_rates, (1 + clip_range) * off_policy_rates) * advantages
        clip_frac = (torch.abs((ratios / off_policy_rates - 1)) > clip_range).float().mean()
        return - torch.min(surr1, surr2).mean(), clip_frac


class TransformerEmbedding(nn.Module):
    def __init__(self):
        super(TransformerEmbedding, self).__init__()

        cfg = Config()

        self.max_num_loops = cfg.max_num_loops
        self.max_num_stores_loads = cfg.max_num_stores_loads
        self.max_num_load_store_dim = cfg.max_num_load_store_dim
        self.op_type_size = len(OperationType)
        self.op_count_size = len(OpFeatures.arith_ops)
        self.loop_access_size = self.max_num_stores_loads * self.max_num_load_store_dim

        self.d_model = cfg.transformer_d_model
        self.pooling = cfg.transformer_pooling
        self.use_action_history_token = cfg.transformer_use_action_history_token

        self.output_size = self.d_model + HardwareFeatures.size()
        if not self.use_action_history_token:
            self.output_size += ActionHistory.size()

        loop_token_size = 2 + (2 * self.loop_access_size) + self.op_type_size + self.op_count_size

        self.summary_proj = nn.Sequential(
            nn.Linear(OpFeatures.size(), self.d_model),
            nn.GELU(),
            nn.Dropout(cfg.transformer_dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        self.loop_proj = nn.Sequential(
            nn.Linear(loop_token_size, self.d_model),
            nn.GELU(),
            nn.Dropout(cfg.transformer_dropout),
            nn.Linear(self.d_model, self.d_model),
        )

        if self.use_action_history_token:
            self.action_history_proj = nn.Sequential(
                nn.Linear(ActionHistory.size(), self.d_model),
                nn.GELU(),
                nn.Dropout(cfg.transformer_dropout),
                nn.Linear(self.d_model, self.d_model),
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # 0: global, 1: consumer, 2: producer
        self.role_embedding = nn.Embedding(3, self.d_model)
        # 0: cls, 1: summary, 2: loop, 3: action-history
        self.token_type_embedding = nn.Embedding(4, self.d_model)
        # depth 0 for global/summary/action-history, 1..L for loop levels
        self.depth_embedding = nn.Embedding(self.max_num_loops + 1, self.d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=cfg.transformer_nhead,
            dim_feedforward=cfg.transformer_ffn_dim,
            dropout=cfg.transformer_dropout,
            activation=cfg.transformer_activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.transformer_num_layers)
        self.out_norm = nn.LayerNorm(self.d_model)

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        return super().__call__(obs)

    def __split_op_features(self, op_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = op_feats.shape[0]
        load_or_store_size = self.loop_access_size * self.max_num_loops

        idx = 0

        op_type = op_feats[:, idx:idx + self.op_type_size]
        idx += self.op_type_size

        loop_upper_bounds = op_feats[:, idx:idx + self.max_num_loops]
        idx += self.max_num_loops

        loop_iterator_types = op_feats[:, idx:idx + self.max_num_loops]
        idx += self.max_num_loops

        load_access = op_feats[:, idx:idx + load_or_store_size].reshape(
            batch_size,
            self.max_num_stores_loads,
            self.max_num_load_store_dim,
            self.max_num_loops,
        )
        idx += load_or_store_size

        store_access = op_feats[:, idx:idx + load_or_store_size].reshape(
            batch_size,
            self.max_num_stores_loads,
            self.max_num_load_store_dim,
            self.max_num_loops,
        )
        idx += load_or_store_size

        op_counts = op_feats[:, idx:idx + self.op_count_size]

        load_access_per_loop = load_access.permute(0, 3, 1, 2).reshape(batch_size, self.max_num_loops, -1)
        store_access_per_loop = store_access.permute(0, 3, 1, 2).reshape(batch_size, self.max_num_loops, -1)

        return op_type, loop_upper_bounds, loop_iterator_types, load_access_per_loop, store_access_per_loop, op_counts

    def __loop_tokens(self, op_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        op_type, loop_upper_bounds, loop_iterator_types, load_access, store_access, op_counts = self.__split_op_features(op_feats)
        batch_size = op_feats.shape[0]

        op_type_expanded = op_type.unsqueeze(1).expand(batch_size, self.max_num_loops, -1)
        op_counts_expanded = op_counts.unsqueeze(1).expand(batch_size, self.max_num_loops, -1)

        token_inputs = torch.cat((
            loop_upper_bounds.unsqueeze(-1),
            loop_iterator_types.unsqueeze(-1),
            load_access,
            store_access,
            op_type_expanded,
            op_counts_expanded,
        ), dim=2)

        # Loop validity must depend only on loop-local signals.
        loop_signal = (
            loop_upper_bounds.abs()
            + loop_iterator_types.abs()
            + load_access.abs().sum(dim=2)
            + store_access.abs().sum(dim=2)
        )
        loop_valid = loop_signal > 0

        return self.loop_proj(token_inputs), loop_valid

    def __add_structure_embeddings(
        self,
        tokens: torch.Tensor,
        role_ids: torch.Tensor,
        type_ids: torch.Tensor,
        depth_ids: torch.Tensor,
    ) -> torch.Tensor:
        return (
            tokens
            + self.role_embedding(role_ids)
            + self.token_type_embedding(type_ids)
            + self.depth_embedding(depth_ids)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]

        consumer_feats = Observation.get_part(obs, OpFeatures)
        producer_feats = Observation.get_part(obs, ProducerOpFeatures)
        action_history = Observation.get_part(obs, ActionHistory)
        hardware_feats = Observation.get_part(obs, HardwareFeatures)
        consumer_num_loops = Observation.get_part(obs, NumLoops).to(dtype=torch.long)

        consumer_loop_tokens, _ = self.__loop_tokens(consumer_feats)
        producer_loop_tokens, producer_loop_valid = self.__loop_tokens(producer_feats)

        consumer_summary = self.summary_proj(consumer_feats).unsqueeze(1)
        producer_summary = self.summary_proj(producer_feats).unsqueeze(1)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        loop_depths = torch.arange(1, self.max_num_loops + 1, device=obs.device).unsqueeze(0).expand(batch_size, -1)
        consumer_loop_valid = loop_depths <= consumer_num_loops.unsqueeze(1)

        token_chunks = [
            cls_tokens,
            consumer_summary,
            producer_summary,
            consumer_loop_tokens,
            producer_loop_tokens,
        ]

        role_chunks = [
            torch.zeros((batch_size, 1), dtype=torch.long, device=obs.device),
            torch.ones((batch_size, 1), dtype=torch.long, device=obs.device),
            torch.full((batch_size, 1), 2, dtype=torch.long, device=obs.device),
            torch.ones((batch_size, self.max_num_loops), dtype=torch.long, device=obs.device),
            torch.full((batch_size, self.max_num_loops), 2, dtype=torch.long, device=obs.device),
        ]

        type_chunks = [
            torch.zeros((batch_size, 1), dtype=torch.long, device=obs.device),
            torch.ones((batch_size, 1), dtype=torch.long, device=obs.device),
            torch.ones((batch_size, 1), dtype=torch.long, device=obs.device),
            torch.full((batch_size, self.max_num_loops), 2, dtype=torch.long, device=obs.device),
            torch.full((batch_size, self.max_num_loops), 2, dtype=torch.long, device=obs.device),
        ]

        depth_chunks = [
            torch.zeros((batch_size, 1), dtype=torch.long, device=obs.device),
            torch.zeros((batch_size, 1), dtype=torch.long, device=obs.device),
            torch.zeros((batch_size, 1), dtype=torch.long, device=obs.device),
            loop_depths,
            loop_depths,
        ]

        valid_chunks = [
            torch.ones((batch_size, 1), dtype=torch.bool, device=obs.device),
            torch.ones((batch_size, 1), dtype=torch.bool, device=obs.device),
            torch.ones((batch_size, 1), dtype=torch.bool, device=obs.device),
            consumer_loop_valid,
            producer_loop_valid,
        ]

        if self.use_action_history_token:
            token_chunks.append(self.action_history_proj(action_history).unsqueeze(1))
            role_chunks.append(torch.zeros((batch_size, 1), dtype=torch.long, device=obs.device))
            type_chunks.append(torch.full((batch_size, 1), 3, dtype=torch.long, device=obs.device))
            depth_chunks.append(torch.zeros((batch_size, 1), dtype=torch.long, device=obs.device))
            valid_chunks.append(torch.ones((batch_size, 1), dtype=torch.bool, device=obs.device))

        tokens = torch.cat(token_chunks, dim=1)
        role_ids = torch.cat(role_chunks, dim=1)
        type_ids = torch.cat(type_chunks, dim=1)
        depth_ids = torch.cat(depth_chunks, dim=1)
        valid_mask = torch.cat(valid_chunks, dim=1)

        tokens = self.__add_structure_embeddings(tokens, role_ids, type_ids, depth_ids)
        encoded = self.transformer(tokens, src_key_padding_mask=~valid_mask)
        encoded = self.out_norm(encoded)

        if self.pooling == 'mean':
            weights = valid_mask.unsqueeze(-1).to(encoded.dtype)
            pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        else:
            pooled = encoded[:, 0, :]

        if self.use_action_history_token:
            result = pooled
        else:
            result = torch.cat((pooled, action_history), dim=1)

        return torch.cat((result, hardware_feats), dim=1)
