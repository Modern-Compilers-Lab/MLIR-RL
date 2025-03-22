import torch
import torch.nn as nn
from torch.distributions import Categorical, Binomial, Normal, Distribution, Uniform
from typing import Optional, Union
from rl_autoschedular import config as cfg
import math


class HiearchyModel(nn.Module):
    """Hierarchical reinforcement learning model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(HiearchyModel, self).__init__()

        N = cfg.num_transformations
        L = cfg.max_num_loops
        D = cfg.max_num_load_store_dim
        SD = cfg.max_num_stores_loads

        match cfg.interchange_mode:
            case 'enumerate':
                interchange_mask = 3 * L - 6
                interchange_input = 0
            case 'pointers':
                interchange_mask = L
                interchange_input = L
            case 'continuous':
                interchange_mask = 0
                interchange_input = 0

        self.input_dim = 5 + L + L * D * SD + L * D + 5 + interchange_input + cfg.truncate * 3 * L
        self.action_mask_size = N + L + L + interchange_mask

        self.policy_model = PolicyModel(self.input_dim, self.action_mask_size)
        self.value_model = ValueModel(self.input_dim, self.action_mask_size)

        if cfg.exploration == 'curiosity':
            self.icm_model = ICMModel(self.input_dim, self.action_mask_size)

    def __call__(self, obs: torch.Tensor, num_loops: list[int], actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(obs, num_loops, actions)

    def forward(self, obs: torch.Tensor, num_loops: list[int], actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        action_log_p, entropy = self.__calculate_dist_stats(
            list(self.policy_model(obs, num_loops)),
            list(raw_actions_to_indices(actions))
        )

        values = self.value_model(obs)

        return action_log_p, values, entropy

    def sample(self, obs: torch.Tensor, num_loops: list[int], greedy: bool = False, eps: Optional[float] = None) -> tuple[list[tuple[str, Optional[Union[list[int], int]]]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the model.

        Args:
            obs (torch.Tensor): The input tensor.
            num_loops (list[int]): The number of loops for each element in the batch.
            actions (Optional[list[tuple[str, Optional[Union[list[int], int]]]]]): list of actions forced for the model to return. Defaults to None.

        Returns:
            Optional[list[tuple[str, Optional[Union[list[int], int]]]]]: list of actions.
            torch.Tensor: action log probabilities.
            torch.Tensor: action values.
            torch.Tensor: resulting entropy.
        """
        assert not greedy or eps is None, 'Cannot be greedy and explore at the same time.'

        # Model feedforward
        transformation_dist, parallelization_dist, tiling_dist, interchange_dist = self.policy_model(obs, num_loops)
        values = self.value_model(obs)

        # Sample actions
        if greedy:
            # Get the indices of the maximum probability
            transformation_index = transformation_dist.probs.argmax(-1)
            parallelization_index = parallelization_dist.probs.argmax(-1)
            tiling_index = tiling_dist.probs.argmax(-1)
            if cfg.interchange_mode == 'continuous':
                interchange_index = interchange_dist.mean.long()
            else:
                interchange_index = interchange_dist.probs.argmax(-1)
        else:
            if eps is not None and torch.rand(1).item() < eps:
                # Sample actions uniformly
                transformation_uni_dist, parallelization_uni_dist, tiling_uni_dist, interchange_uni_dist = self.__create_uniform_distributions(obs, num_loops)
            else:
                transformation_uni_dist, parallelization_uni_dist, tiling_uni_dist, interchange_uni_dist = transformation_dist, parallelization_dist, tiling_dist, interchange_dist
            # Sample actions
            transformation_index = transformation_uni_dist.sample()
            parallelization_index = parallelization_uni_dist.sample()
            tiling_index = tiling_uni_dist.sample()
            interchange_index = interchange_uni_dist.sample().long()

        if cfg.interchange_mode == 'continuous':
            # Clamp interchange index to [0, num_loops! - 1]
            total_count = torch.tensor([math.factorial(loops) for loops in num_loops], dtype=torch.float64)
            interchange_index = interchange_index.clamp(torch.zeros_like(total_count), total_count - 1)

        # Get raw actions from indices
        actions = indices_to_raw_actions(transformation_index, parallelization_index, tiling_index, interchange_index, num_loops)

        # Calculate the log probabilities and entropies
        action_log_p, entropy = self.__calculate_dist_stats(
            [transformation_dist, parallelization_dist, tiling_dist, interchange_dist],
            [transformation_index, parallelization_index, tiling_index, interchange_index]
        )

        return actions, action_log_p, values, entropy

    def __create_uniform_distributions(self, obs: torch.Tensor, num_loops: list[int]) -> tuple[Distribution, Distribution, Distribution, Distribution]:
        """Create uniform distributions for the actions.

        Args:
            obs (torch.Tensor): The input tensor.

        Returns:
            tuple[Distribution, Distribution, Distribution, Distribution]: The uniform distributions for the transformations, parallelizations, tilings, and interchanges.
        """
        N = cfg.num_transformations
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        batch_size = obs.shape[0]
        action_mask = obs[:, -(self.action_mask_size):].bool()

        transformation_logits = torch.zeros((batch_size, N), dtype=torch.float32)
        parallelization_logits = torch.zeros((batch_size, L, TS + 1), dtype=torch.float32)
        tiling_logits = torch.zeros((batch_size, L, TS + 1), dtype=torch.float32)
        match cfg.interchange_mode:
            case 'enumerate':
                interchange_logits = torch.zeros((batch_size, 3 * L - 6), dtype=torch.float32)
            case 'pointers':
                interchange_logits = torch.zeros((batch_size, L), dtype=torch.float32)
            case 'continuous':
                interchange_logits = torch.zeros((batch_size, 1), dtype=torch.float32)

        # Apply masks on logits
        transformation_logits, parallelization_logits, tiling_logits, interchange_logits = apply_masks(transformation_logits, parallelization_logits, tiling_logits, interchange_logits, *extract_masks(action_mask))

        # Create distributions with the masked probabilities
        transformation_dist = Categorical(logits=transformation_logits)
        parallelization_dist = Categorical(logits=parallelization_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        if cfg.interchange_mode != 'continuous':
            interchange_dist = Categorical(logits=interchange_logits)
        else:
            total_count = torch.tensor([math.factorial(loops) for loops in num_loops], dtype=torch.float64)
            interchange_dist = Uniform(0.0, total_count)

        return transformation_dist, parallelization_dist, tiling_dist, interchange_dist

    def __calculate_dist_stats(self, dists: list[Distribution], indices: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the log probabilities and entropies of the actions.

        Args:
            transformation_dist (Distribution): The transformation distribution.
            parallelization_dist (Distribution): The parallelization distribution.
            tiling_dist (Distribution): The tiling distribution.
            interchange_dist (Distribution): The interchange distribution.
            transformation_index (torch.Tensor): The transformation indices.
            parallelization_index (torch.Tensor): The parallelization indices.
            tiling_index (torch.Tensor): The tiling indices.
            interchange_index (torch.Tensor): The interchange indices.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The log probabilities and entropies of the
        """
        transformation_dist, parallelization_dist, tiling_dist, interchange_dist = dists
        transformation_index, parallelization_index, tiling_index, interchange_index = indices

        batch_size = transformation_index.shape[0]

        transformation_log_p = transformation_dist.log_prob(transformation_index)
        parallelization_log_p = parallelization_dist.log_prob(parallelization_index).sum(-1)
        tiling_log_p = tiling_dist.log_prob(tiling_index).sum(-1)
        interchange_log_p = interchange_dist.log_prob(interchange_index)
        if isinstance(interchange_dist, Normal):
            # Ensure that the log probability is not very small
            interchange_log_p = interchange_log_p.maximum(torch.tensor(-100.0))

        # Calculate the total log probability and entropy
        action_log_p = transformation_log_p
        entropy = transformation_dist.entropy()
        if cfg.mul_log_p:
            action_log_p += parallelization_log_p * (transformation_index == 1)
            action_log_p += tiling_log_p * (transformation_index == 2)
            action_log_p += interchange_log_p * (transformation_index == 3)
            entropy += parallelization_dist.entropy().sum(-1) * (transformation_index == 1)
            entropy += tiling_dist.entropy().sum(-1) * (transformation_index == 2)
            if not isinstance(interchange_dist, Binomial) or batch_size == 1:
                entropy += interchange_dist.entropy() * (transformation_index == 3)
        else:
            action_log_p[transformation_index == 1] += parallelization_log_p[transformation_index == 1]
            action_log_p[transformation_index == 2] += tiling_log_p[transformation_index == 2]
            action_log_p[transformation_index == 3] += interchange_log_p[transformation_index == 3]
            entropy[transformation_index == 1] += parallelization_dist.entropy().sum(-1)[transformation_index == 1]
            entropy[transformation_index == 2] += tiling_dist.entropy().sum(-1)[transformation_index == 2]
            if not isinstance(interchange_dist, Binomial) or batch_size == 1:
                entropy[transformation_index == 3] += interchange_dist.entropy()[transformation_index == 3]

        return action_log_p, entropy


class ValueModel(nn.Module):
    """Value model for MLIR code optimization."""
    def __init__(self, input_dim: int, action_mask_size: int):
        """Initialize the model.

        Args:
            input_dim (int): The input dimension.
        """
        super(ValueModel, self).__init__()

        self.input_dim = input_dim
        self.action_mask_size = action_mask_size
        activation_layer = nn.ReLU if cfg.activation == 'relu' else nn.Tanh
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
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
        return self.network(obs[:, :-(self.action_mask_size)])

    def loss(self, new_values: torch.Tensor, values: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        """Calculate the value loss.

        Args:
            new_values (torch.Tensor): The new value tensor.
            values (torch.Tensor): The value tensor.
            returns (torch.Tensor): The returns tensor.

        Returns:
            torch.Tensor: The value loss.
        """
        vclip = values + torch.clamp(new_values - values, -0.2, 0.2)
        vloss1 = (returns - vclip).pow(2)
        vloss2 = (returns - new_values).pow(2)
        return 0.5 * torch.max(vloss1, vloss2).mean()


class PolicyModel(nn.Module):
    """Policy model for MLIR code optimization."""
    def __init__(self, input_dim: int, action_mask_size: int):
        """Initialize the model.

        Args:
            input_dim (int): The input dimension.
        """
        super(PolicyModel, self).__init__()

        self.input_dim = input_dim
        self.action_mask_size = action_mask_size
        activation_layer = nn.ReLU if cfg.activation == 'relu' else nn.Tanh
        N = cfg.num_transformations
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes

        match cfg.interchange_mode:
            case 'enumerate':
                interchange_layer = nn.Linear(512, 3 * L - 6)
            case 'pointers':
                interchange_layer = nn.Linear(512, L)
            case 'continuous':
                interchange_layer = nn.Linear(512, 1)
                if cfg.interchange_distribution == 'normal':
                    self.interchange_logstd = nn.Parameter(torch.zeros(1))

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
        )

        self.transformation_fc = nn.Linear(512, N)
        self.parallelization_fc = nn.Linear(512, L * (TS + 1))
        self.tiling_fc = nn.Linear(512, L * (TS + 1))
        self.interchange_fc = interchange_layer

        if cfg.new_architecture:
            self.transformation_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                self.transformation_fc,
            )

            self.parallelization_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                self.parallelization_fc,
            )

            self.tiling_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                self.tiling_fc,
            )

            self.interchange_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                self.interchange_fc,
            )

    def __call__(self, obs: torch.Tensor, num_loops: list[int]) -> tuple[Distribution, Distribution, Distribution, Distribution]:
        return super().__call__(obs, num_loops)

    def forward(self, obs: torch.Tensor, num_loops: list[int]) -> tuple[Distribution, Distribution, Distribution, Distribution]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        batch_size = obs.shape[0]

        x = obs[:, :-(self.action_mask_size)]
        action_mask = obs[:, -(self.action_mask_size):].bool()

        # Model feedforward
        x = self.backbone(x)

        transformation_logits = self.transformation_fc(x)
        parallelization_logits = self.parallelization_fc(x).reshape(batch_size, L, TS + 1)
        tiling_logits = self.tiling_fc(x).reshape(batch_size, L, TS + 1)
        interchange_logits = self.interchange_fc(x)

        # Apply masks on logits
        transformation_logits, parallelization_logits, tiling_logits, interchange_logits = apply_masks(transformation_logits, parallelization_logits, tiling_logits, interchange_logits, *extract_masks(action_mask))

        # Create distributions with the masked probabilities
        transformation_dist = Categorical(logits=transformation_logits)
        parallelization_dist = Categorical(logits=parallelization_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        if cfg.interchange_mode != 'continuous':
            interchange_dist = Categorical(logits=interchange_logits)
        else:
            interchange_logit = interchange_logits.squeeze(-1)
            if cfg.interchange_distribution == 'binomial':
                total_count = torch.tensor([math.factorial(loops) for loops in num_loops])
                interchange_dist = Binomial(total_count, logits=interchange_logit)
            else:
                interchange_dist = Normal(interchange_logit, self.interchange_logstd.clamp(-1, 1).exp())

        return transformation_dist, parallelization_dist, tiling_dist, interchange_dist

    def loss(self, new_actions_log_p: torch.Tensor, actions_log_p: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
        """Calculate the policy loss.

        Args:
            new_actions_log_p (torch.Tensor): The log probabilities of the new actions.
            actions_log_p (torch.Tensor): The log probabilities of the actions.
            advantages (torch.Tensor): The advantages of the actions.

        Returns:
            torch.Tensor: The policy loss.
        """
        ratios = torch.exp(new_actions_log_p - actions_log_p)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - 0.2, 1 + 0.2) * advantages
        return - torch.min(surr1, surr2).mean()


class ICMModel(nn.Module):
    """Inverse Curiosity Model for MLIR code optimization."""
    def __init__(self, input_dim: int, action_mask_size: int):
        """Initialize the model.

        Args:
            input_dim (int): The input dimension.
        """
        super(ICMModel, self).__init__()

        self.input_dim = input_dim
        self.action_mask_size = action_mask_size
        activation_layer = nn.ReLU if cfg.activation == 'relu' else nn.Tanh

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
            nn.Linear(512, 512),
        )

        self.forward_model = ForwardModel()
        self.inverse_model = InverseModel()

    def __call__(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        return super().__call__(obs, next_obs, actions)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor, actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the model.

        Args:
            obs (torch.Tensor): The input tensor.
            next_obs (torch.Tensor): The next input tensor.
            actions (list[tuple[str, Optional[Union[list[int], int]]]]): The list of actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        x = obs[:, :-(self.action_mask_size)]
        next_x = next_obs[:, :-(self.action_mask_size)]
        action_mask = obs[:, -(self.action_mask_size):].bool()

        state_latent = self.encoder(x)
        next_state_latent = self.encoder(next_x)

        next_state_latent_hat = self.forward_model(state_latent, actions)
        action_logits_hat = self.inverse_model(state_latent, next_state_latent, action_mask)

        return next_state_latent, next_state_latent_hat, action_logits_hat

    def loss(self, next_states_latent: torch.Tensor, next_states_latent_hat: torch.Tensor, action_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> torch.Tensor:
        """Calculate the ICM loss.

        Args:
            next_states_latent (torch.Tensor): The next latent state tensor.
            next_states_latent_hat (torch.Tensor): The predicted next latent state tensor.
            action_logits (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The predicted logits of the actions.
            actions (list[tuple[str, Optional[Union[list[int], int]]]): The list of actions.

        Returns:
            torch.Tensor: The ICM loss.
        """
        return cfg.forward_weight * self.forward_model.loss(next_states_latent, next_states_latent_hat) + (1 - cfg.forward_weight) * self.inverse_model.loss(action_logits, actions)


class ForwardModel(nn.Module):
    """Forward model for Inverse Curiosity Model."""
    def __init__(self):
        """Initialize the model."""
        super(ForwardModel, self).__init__()

        N = cfg.num_transformations
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        activation_layer = nn.ReLU if cfg.activation == 'relu' else nn.Tanh

        self.transformation_encoder = nn.Embedding(N, 8)
        self.parallelization_encoder = nn.Embedding(TS + 1, 8)
        self.tiling_encoder = nn.Embedding(TS + 1, 8)
        match cfg.interchange_mode:
            case 'enumerate':
                self.interchange_encoder = nn.Embedding(3 * L - 6, 8)
            case 'pointers':
                self.interchange_encoder = nn.Embedding(L, 8)
            case 'continuous':
                self.interchange_encoder = nn.Linear(1, 8)
        self.action_encoder = nn.Sequential(
            nn.Linear(16 * (L + 1), 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
            nn.Linear(512, 512),
        )

        self.network = nn.Sequential(
            nn.Linear(512 + 512, 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
            nn.Linear(512, 512),
        )

    def __call__(self, state_latent: torch.Tensor, actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> torch.Tensor:
        return super().__call__(state_latent, actions)

    def forward(self, state_latent: torch.Tensor, actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            state_latent (torch.Tensor): The latent state tensor.
            actions (list[tuple[str, Optional[Union[list[int], int]]]): The list of actions.

        Returns:
            torch.Tensor: The predicted latent state tensor.
        """
        batch_size = state_latent.shape[0]

        transformation_index, parallelization_index, tiling_index, interchange_index = raw_actions_to_indices(actions)

        transformation_latent = self.transformation_encoder(transformation_index)
        parallelization_latent = self.parallelization_encoder(parallelization_index).reshape(batch_size, -1)
        tiling_latent = self.tiling_encoder(tiling_index).reshape(batch_size, -1)
        interchange_latent = self.interchange_encoder(interchange_index)

        action_latent = torch.cat((transformation_latent, parallelization_latent, tiling_latent, interchange_latent), dim=-1)

        action_latent = self.action_encoder(action_latent)

        x = torch.cat((action_latent, state_latent), dim=-1)
        x = self.network(x)

        return x

    def loss(self, next_states_latent: torch.Tensor, next_states_latent_hat: torch.Tensor) -> torch.Tensor:
        """Calculate the forward model loss.

        Args:
            next_states_latent (torch.Tensor): The next latent state tensor.
            next_states_latent_hat (torch.Tensor): The predicted next latent state tensor.

        Returns:
            torch.Tensor: The forward model loss.
        """
        return 0.5 * (next_states_latent_hat - next_states_latent).norm(2, dim=-1).pow(2).mean()


class InverseModel(nn.Module):
    """Inverse model for Inverse Curiosity Model."""
    def __init__(self):
        """Initialize the model.

        Args:
            input_dim (int): The input dimension.
        """
        super(InverseModel, self).__init__()
        self.disc_loss = nn.CrossEntropyLoss(reduction='none')
        self.cont_loss = nn.MSELoss(reduction='none')

        N = cfg.num_transformations
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        activation_layer = nn.ReLU if cfg.activation == 'relu' else nn.Tanh

        match cfg.interchange_mode:
            case 'enumerate':
                interchange_layer = nn.Linear(512, 3 * L - 6)
            case 'pointers':
                interchange_layer = nn.Linear(512, L)
            case 'continuous':
                interchange_layer = nn.Linear(512, 1)

        self.backbone = nn.Sequential(
            nn.Linear(512 * 2, 512),
            activation_layer(),
            nn.Linear(512, 512),
            activation_layer(),
            nn.Linear(512, 512),
        )

        self.transformation_fc = nn.Linear(512, N)
        self.parallelization_fc = nn.Linear(512, L * (TS + 1))
        self.tiling_fc = nn.Linear(512, L * (TS + 1))
        self.interchange_fc = interchange_layer

    def __call__(self, state_latent: torch.Tensor, next_state_latent: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(state_latent, next_state_latent, action_mask)

    def forward(self, state_latent: torch.Tensor, next_state_latent: torch.Tensor, action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            state_latent (torch.Tensor): The latent state tensor.
            next_state_latent (torch.Tensor): The next latent state tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        batch_size = state_latent.shape[0]

        x = torch.cat((state_latent, next_state_latent), dim=-1)

        x = self.backbone(x)

        transformation_logits = self.transformation_fc(x)
        parallelization_logits = self.parallelization_fc(x).reshape(batch_size, L, TS + 1)
        tiling_logits = self.tiling_fc(x).reshape(batch_size, L, TS + 1)
        interchange_logits = self.interchange_fc(x)

        return apply_masks(transformation_logits, parallelization_logits, tiling_logits, interchange_logits, *extract_masks(action_mask))

    def loss(self, action_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> torch.Tensor:
        """Calculate the inverse model loss.

        Args:
            action_logits (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The predicted logits of the actions.
            actions (list[tuple[str, Optional[Union[list[int], int]]]): The list of actions.

        Returns:
            torch.Tensor: The inverse model loss.
        """
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        batch_size = len(actions)

        transformation_index, parallelization_index, tiling_index, interchange_index = raw_actions_to_indices(actions)
        transformation_logits_hat, parallelization_logits_hat, tiling_logits_hat, interchange_logits_hat = action_logits

        transformation_loss = self.disc_loss(transformation_logits_hat, transformation_index)
        parallelization_loss = self.disc_loss(parallelization_logits_hat.reshape(batch_size * L, TS + 1), parallelization_index.flatten()).reshape(batch_size, L).sum(-1)
        tiling_loss = self.disc_loss(tiling_logits_hat.reshape(batch_size * L, TS + 1), tiling_index.flatten()).reshape(batch_size, L).sum(-1)
        if cfg.interchange_mode == 'continuous':
            interchange_loss = self.cont_loss(interchange_logits_hat.squeeze(-1), interchange_index.float())
        else:
            interchange_loss = self.disc_loss(interchange_logits_hat, interchange_index)

        loss = transformation_loss
        if cfg.mul_log_p:
            loss += parallelization_loss * (transformation_index == 1)
            loss += tiling_loss * (transformation_index == 2)
            loss += interchange_loss * (transformation_index == 3)
        else:
            loss[transformation_index == 1] += parallelization_loss[transformation_index == 1]
            loss[transformation_index == 2] += tiling_loss[transformation_index == 2]
            loss[transformation_index == 3] += interchange_loss[transformation_index == 3]

        return loss.mean()


def extract_masks(action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract masks from the action mask tensor.

    Args:
        action_mask (torch.Tensor): The action mask tensor.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: The masks for the transformations, parallelizations, tilings, and interchanges.
    """
    batch_size = action_mask.shape[0]
    L = cfg.max_num_loops
    TS = cfg.num_tile_sizes
    TP_BEGIN = cfg.num_transformations
    T_BEGIN = TP_BEGIN + L
    I_BEGIN = T_BEGIN + L

    transform_mask = action_mask[:, :cfg.num_transformations]
    TP_mask = action_mask[:, TP_BEGIN:T_BEGIN].unsqueeze(-1).broadcast_to(batch_size, L, TS + 1)
    TP_mask[:, :, 0] = 1  # Never mask "no tiling"
    T_mask = action_mask[:, T_BEGIN:I_BEGIN].unsqueeze(-1).broadcast_to(batch_size, L, TS + 1)
    T_mask[:, :, 0] = 1  # Never mask "no tiling"
    if cfg.interchange_mode == 'continuous':
        I_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    else:
        I_mask = action_mask[:, I_BEGIN:]

    return transform_mask, TP_mask, T_mask, I_mask


def apply_masks(*args: torch.Tensor, value: float = -torch.inf) -> list[torch.Tensor]:
    """Apply masks to the logits tensors.

    Args:
        args (torch.Tensor): The logits tensors followed by the action mask tensors.

    Returns:
        torch.Tensor: The masked logits tensors.
    """
    args_count = len(args)
    assert args_count % 2 == 0, 'The number of arguments must be even.'
    logits = args[:args_count // 2]
    action_masks = args[args_count // 2:]
    masked_logits = []
    for logit, action_mask in zip(logits, action_masks):
        masked_logits.append(logit.where(action_mask, value))

    return masked_logits


def transformation_to_int(transformation: str) -> int:
    """Convert a transformation string to an integer.

    Args:
        transformation (str): The transformation string.

    Returns:
        int: The transformation integer.
    """
    return {
        'no_transformation': 0,
        'parallelization': 1,
        'tiling': 2,
        'interchange': 3,
        'vectorization': 4,
    }[transformation]


def int_to_transformation(transformation: int) -> str:
    """Convert an integer to a transformation string.

    Args:
        transformation (int): The transformation integer.

    Returns:
        str: The transformation string.
    """
    return {
        0: 'no_transformation',
        1: 'parallelization',
        2: 'tiling',
        3: 'interchange',
        4: 'vectorization',
    }[transformation]


def raw_actions_to_indices(actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a list of actions to tensor of indices.

    Args:
        actions (Optional[list[tuple[str, Optional[Union[list[int], int]]]]): The list of actions.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The indices tensors for the transformations, parallelizations, tilings, and interchanges.
    """
    L = cfg.max_num_loops
    batch_size = len(actions)
    transformation_index = torch.tensor([transformation_to_int(name) for name, _ in actions], dtype=torch.int64)

    parallelization_index = torch.zeros((batch_size, L), dtype=torch.int64)
    tiling_index = torch.zeros((batch_size, L), dtype=torch.int64)
    interchange_index = torch.zeros((batch_size,), dtype=torch.int64)
    for i, action in enumerate(actions):
        action_name, parameters = action
        match action_name:
            case 'parallelization':
                parallelization_index[i, :len(parameters)] = torch.tensor(parameters)
            case 'tiling':
                tiling_index[i, :len(parameters)] = torch.tensor(parameters)
            case 'interchange':
                interchange_index[i] = parameters

    return transformation_index, parallelization_index, tiling_index, interchange_index


def indices_to_raw_actions(transformation_index: torch.Tensor, parallelization_index: torch.Tensor, tiling_index: torch.Tensor, interchange_index: torch.Tensor, num_loops: list[int]) -> list[tuple[str, Optional[Union[list[int], int]]]]:
    """Convert tensor indices to a list of actions.

    Args:
        transformation_index (torch.Tensor): The transformation indices.
        parallelization_index (torch.Tensor): The parallelization indices.
        tiling_index (torch.Tensor): The tiling indices.
        interchange_index (torch.Tensor): The interchange indices.

    Returns:
        list[tuple[str, Optional[Union[list[int], int]]]]: The list of actions.
    """
    actions = []
    for i in range(transformation_index.shape[0]):
        transformation = int_to_transformation(transformation_index[i].item())
        parameters = None
        match transformation:
            case 'parallelization':
                parameters = parallelization_index[i, :num_loops[i]].tolist()
            case 'tiling':
                parameters = tiling_index[i, :num_loops[i]].tolist()
            case 'interchange':
                parameters = interchange_index[i].item()
        actions.append((transformation, parameters))

    return actions
