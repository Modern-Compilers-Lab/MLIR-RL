import torch
import torch.nn as nn
from torch.distributions import Categorical, Binomial, Normal, Distribution, Uniform
from typing import Optional, Union
from rl_autoschedular import config as cfg
from rl_autoschedular.observation import build_tree


class HiearchyModel(nn.Module):
    """Hierarchical reinforcement learning model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(HiearchyModel, self).__init__()

        N = cfg.num_transformations
        L = cfg.max_num_loops
        D = cfg.max_num_load_store_dim
        SD = cfg.max_num_stores_loads
        TS = cfg.num_tile_sizes

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

        embedding_size = 411
        op_feats_size = 5 + L + L + 1 + L * D * SD + L * D + 5
        action_history_size = interchange_input + cfg.truncate * 4 * L
        self.action_mask_size = N + 3 * L * (TS + 1) + interchange_mask

        self.policy_model = PolicyModel(embedding_size, op_feats_size, action_history_size, self.action_mask_size)
        self.value_model = ValueModel(embedding_size, op_feats_size, action_history_size, self.action_mask_size)

    def __call__(self, obs: torch.Tensor, num_loops: torch.Tensor, actions_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().__call__(obs, num_loops, actions_index)

    def forward(self, obs: torch.Tensor, num_loops: torch.Tensor, actions_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        action_log_p, entropy = self.__calculate_dist_stats(
            list(self.policy_model(obs, num_loops)),
            list(decode_actions_index(actions_index))
        )

        values = self.value_model(obs)

        return action_log_p, values, entropy

    def sample(self, obs: torch.Tensor, num_loops: torch.Tensor, greedy: bool = False, eps: Optional[float] = None) -> tuple[list[tuple[str, Optional[Union[list[int], int]]]], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the model.

        Args:
            obs (torch.Tensor): The input tensor.
            num_loops (torch.Tensor): The number of loops for each element in the batch.

        Returns:
            list[tuple[str, Optional[Union[list[int], int]]]]: list of actions.
            torch.Tensor: actions log probability.
            torch.Tensor: actions value.
            torch.Tensor: resulting entropy.
        """
        assert not greedy or eps is None, 'Cannot be greedy and explore at the same time.'

        # Model feedforward
        transformation_dist, parallelization_dist, tiling_dist, interchange_dist, fusion_dist = self.policy_model(obs, num_loops)
        values = self.value_model(obs)

        # Sample actions
        if greedy:
            # Get the indices of the maximum probability
            transformation_index = transformation_dist.probs.argmax(-1)
            parallelization_index = parallelization_dist.probs.argmax(-1)
            tiling_index = tiling_dist.probs.argmax(-1)
            fusion_index = fusion_dist.probs.argmax(-1)
            if cfg.interchange_mode == 'continuous':
                interchange_index = interchange_dist.mean.long()
            else:
                interchange_index = interchange_dist.probs.argmax(-1)
        else:
            if eps is not None:
                transformation_eps_dist, parallelization_eps_dist, tiling_eps_dist, interchange_eps_dist, fusion_eps_dist = self.__create_uniform_distributions(obs, num_loops)
            if eps is not None and torch.rand(1).item() < eps:
                # Sample actions uniformly
                transformation_index = transformation_eps_dist.sample()
                parallelization_index = parallelization_eps_dist.sample()
                tiling_index = tiling_eps_dist.sample()
                interchange_index = interchange_eps_dist.sample().long()
                fusion_index = fusion_eps_dist.sample()
            else:
                # Sample actions
                transformation_index = transformation_dist.sample()
                parallelization_index = parallelization_dist.sample()
                tiling_index = tiling_dist.sample()
                interchange_index = interchange_dist.sample().long()
                fusion_index = fusion_dist.sample()

        if cfg.interchange_mode == 'continuous':
            # Clamp interchange index to [0, num_loops! - 1]
            total_count = (num_loops + 1).lgamma().exp().long()
            interchange_index = interchange_index.clamp(torch.zeros_like(total_count, dtype=torch.int64), total_count - 1)

        # Get raw actions from indices
        actions = indices_to_raw_actions(transformation_index, parallelization_index, tiling_index, interchange_index, fusion_index, num_loops)
        actions_index = encode_actions_index(transformation_index, parallelization_index, tiling_index, interchange_index, fusion_index)

        # Calculate the log probabilities and entropies
        action_log_p, entropy = self.__calculate_dist_stats(
            [transformation_dist, parallelization_dist, tiling_dist, interchange_dist, fusion_dist],
            [transformation_index, parallelization_index, tiling_index, interchange_index, fusion_index],
            eps_dists=[transformation_eps_dist, parallelization_eps_dist, tiling_eps_dist, interchange_eps_dist, fusion_eps_dist] if eps is not None else None,
            eps=eps
        )

        return actions, actions_index, action_log_p, values, entropy

    def __create_uniform_distributions(self, obs: torch.Tensor, num_loops: torch.Tensor) -> tuple[Distribution, Distribution, Distribution, Distribution, Distribution]:
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
        fusion_logits = torch.zeros((batch_size, L, TS + 1), dtype=torch.float32)
        match cfg.interchange_mode:
            case 'enumerate':
                interchange_logits = torch.zeros((batch_size, 3 * L - 6), dtype=torch.float32)
            case 'pointers':
                interchange_logits = torch.zeros((batch_size, L), dtype=torch.float32)
            case 'continuous':
                interchange_logits = torch.zeros((batch_size, 1), dtype=torch.float32)

        # Apply masks on logits
        transformation_logits, parallelization_logits, tiling_logits, interchange_logits, fusion_logits = apply_masks(transformation_logits, parallelization_logits, tiling_logits, interchange_logits, fusion_logits, *extract_masks(action_mask))

        # Create distributions with the masked probabilities
        transformation_dist = Categorical(logits=transformation_logits)
        parallelization_dist = Categorical(logits=parallelization_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        fusion_dist = Categorical(logits=fusion_logits)
        if cfg.interchange_mode != 'continuous':
            interchange_dist = Categorical(logits=interchange_logits)
        else:
            total_count = (num_loops + 1).lgamma().exp()
            interchange_dist = Uniform(0.0, total_count)

        return transformation_dist, parallelization_dist, tiling_dist, interchange_dist, fusion_dist

    def __calculate_dist_stats(self, dists: list[Distribution], indices: list[torch.Tensor], eps_dists: Optional[list[Distribution]] = None, eps: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
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
            torch.Tensor: The log probabilities
            torch.Tensor: The entropies
        """
        assert (eps_dists is None) == (eps is None), 'eps_dists and eps must be both None or both not None.'

        transformation_dist, parallelization_dist, tiling_dist, interchange_dist, fusion_dist = dists
        transformation_index, parallelization_index, tiling_index, interchange_index, fusion_index = indices

        batch_size = transformation_index.shape[0]

        transformation_log_p = transformation_dist.log_prob(transformation_index)
        parallelization_log_p = parallelization_dist.log_prob(parallelization_index).sum(-1)
        tiling_log_p = tiling_dist.log_prob(tiling_index).sum(-1)
        if isinstance(interchange_dist, Normal):
            # Special case in Normal distribution we need to consider all
            # the interval [i,i+1), so we use log CDF instead of log P
            interchange_log_p = (interchange_dist.cdf(interchange_index + 1) - interchange_dist.cdf(interchange_index)).log()
        else:
            interchange_log_p = interchange_dist.log_prob(interchange_index)
        fusion_log_p = fusion_dist.log_prob(fusion_index).sum(-1)

        if eps_dists is not None:
            transformation_eps_dist, parallelization_eps_dist, tiling_eps_dist, interchange_eps_dist, fusion_eps_dist = eps_dists
            transformation_log_p = ((1 - eps) * transformation_log_p.exp() + eps * transformation_eps_dist.log_prob(transformation_index).exp()).log()
            parallelization_log_p = ((1 - eps) * parallelization_log_p.exp() + eps * parallelization_eps_dist.log_prob(parallelization_index).sum(-1).exp()).log()
            tiling_log_p = ((1 - eps) * tiling_log_p.exp() + eps * tiling_eps_dist.log_prob(tiling_index).sum(-1).exp()).log()
            interchange_log_p = ((1 - eps) * interchange_log_p.exp() + eps * interchange_eps_dist.log_prob(interchange_index).exp()).log()
            fusion_log_p = ((1 - eps) * fusion_log_p.exp() + eps * fusion_eps_dist.log_prob(fusion_index).sum(-1).exp()).log()

        # Calculate the total log probability
        action_log_p = transformation_log_p
        action_log_p[transformation_index == 1] += parallelization_log_p[transformation_index == 1]
        action_log_p[transformation_index == 2] += tiling_log_p[transformation_index == 2]
        action_log_p[transformation_index == 3] += interchange_log_p[transformation_index == 3]
        action_log_p[transformation_index == 5] += fusion_log_p[transformation_index == 5]

        # Calculate the entropy
        entropy = transformation_dist.entropy()
        entropy[transformation_index == 1] += parallelization_dist.entropy().sum(-1)[transformation_index == 1]
        entropy[transformation_index == 2] += tiling_dist.entropy().sum(-1)[transformation_index == 2]
        entropy[transformation_index == 5] += fusion_dist.entropy().sum(-1)[transformation_index == 5]
        if not isinstance(interchange_dist, Binomial) or batch_size == 1:
            entropy[transformation_index == 3] += interchange_dist.entropy()[transformation_index == 3]

        return action_log_p, entropy


class ValueModel(nn.Module):
    """Value model for MLIR code optimization."""
    def __init__(self, embedding_size: int, op_feats_size: int, action_history_size: int, action_mask_size: int):
        """Initialize the model.

        Args:
            input_dim (int): The input dimension.
        """
        super(ValueModel, self).__init__()

        self.action_mask_size = action_mask_size
        activation_layer = nn.ReLU if cfg.activation == 'relu' else nn.Tanh

        self.lstm = LSTMEmbedding(output_size=embedding_size)

        self.network = nn.Sequential(
            nn.Linear(embedding_size + action_history_size, 512),
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
        obs = self.lstm(obs)
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
    def __init__(self, embedding_size: int, op_feats_size: int, action_history_size: int, action_mask_size: int):
        """Initialize the model.

        Args:
            input_dim (int): The input dimension.
        """
        super(PolicyModel, self).__init__()

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

        self.lstm = LSTMEmbedding(output_size=embedding_size)

        self.backbone = nn.Sequential(
            nn.Linear(embedding_size + action_history_size, 512),
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
        self.fusion_fc = nn.Linear(512, L * (TS + 1))

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

            self.fusion_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                self.fusion_fc,
            )

    def __call__(self, obs: torch.Tensor, num_loops: torch.Tensor) -> tuple[Distribution, Distribution, Distribution, Distribution, Distribution]:
        return super().__call__(obs, num_loops)

    def forward(self, obs: torch.Tensor, num_loops: torch.Tensor) -> tuple[Distribution, Distribution, Distribution, Distribution, Distribution]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        batch_size = obs.shape[0]

        obs = self.lstm(obs)

        x = obs[:, :-(self.action_mask_size)]
        action_mask = obs[:, -(self.action_mask_size):].bool()

        # Model feedforward
        x = self.backbone(x)

        transformation_logits = self.transformation_fc(x)
        parallelization_logits = self.parallelization_fc(x).reshape(batch_size, L, TS + 1)
        tiling_logits = self.tiling_fc(x).reshape(batch_size, L, TS + 1)
        fusion_logits = self.fusion_fc(x).reshape(batch_size, L, TS + 1)
        interchange_logits = self.interchange_fc(x)

        # Apply masks on logits
        transformation_logits, parallelization_logits, tiling_logits, interchange_logits, fusion_logits = apply_masks(transformation_logits, parallelization_logits, tiling_logits, interchange_logits, fusion_logits, *extract_masks(action_mask))

        # Create distributions with the masked probabilities
        transformation_dist = Categorical(logits=transformation_logits)
        parallelization_dist = Categorical(logits=parallelization_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        fusion_dist = Categorical(logits=fusion_logits)

        if cfg.interchange_mode != 'continuous':
            interchange_dist = Categorical(logits=interchange_logits)
        else:
            interchange_logit = interchange_logits.squeeze(-1)
            if cfg.interchange_distribution == 'binomial':
                total_count = (num_loops + 1).lgamma().exp().long()
                interchange_dist = Binomial(total_count, logits=interchange_logit)
            else:
                interchange_dist = Normal(interchange_logit, self.interchange_logstd.clamp(-1, 1).exp())

        return transformation_dist, parallelization_dist, tiling_dist, interchange_dist, fusion_dist

    def loss(self, new_actions_log_p: torch.Tensor, mu_actions_log_p: torch.Tensor, off_policy_rates: torch.Tensor, advantages: torch.Tensor, clip_range: float = 0.2) -> torch.Tensor:
        """Calculate the policy loss.

        Args:
            new_actions_log_p (torch.Tensor): The log probabilities of the new actions.
            mu_actions_log_p (torch.Tensor): The log probabilities of the actions under the behavior policy.
            off_policy_rates (torch.Tensor): The rate between the old policy and the behavioral (mu) policy.
            advantages (torch.Tensor): The advantages of the actions.
            clip_range (float): The clipping range for the policy loss.

        Returns:
            torch.Tensor: The policy loss.
        """
        ratios = torch.exp(torch.clamp(new_actions_log_p - mu_actions_log_p, -80.0, 80.0))
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, (1 - clip_range) * off_policy_rates, (1 + clip_range) * off_policy_rates) * advantages
        return - torch.min(surr1, surr2).mean()


def initialization_function_xavier(x):
    return nn.init.xavier_uniform_(x)


class LSTMEmbedding(nn.Module):
    def __init__(self, embedding_size: int, op_feats_size: int, action_history_size: int, action_mask_size: int):
        super(LSTMEmbedding, self).__init__()

        self.op_feats_size = op_feats_size

        self.comp_embed_layer_sizes = [600, 350, 512, 512]

        self.lstm = nn.LSTM(
            self.comp_embed_layer_sizes[-1],
            embedding_size,
            batch_first=True
        )

        self.comps_lstm = nn.LSTM(
            412, embedding_size, batch_first=True
        )
        self.nodes_lstm = nn.LSTM(
            embedding_size, embedding_size, batch_first=True
        )

        self.no_comps_tensor = nn.Parameter(
            initialization_function_xavier(torch.randn(1, embedding_size))
        )
        self.no_nodes_tensor = nn.Parameter(
            initialization_function_xavier(torch.randn(1, embedding_size))
        )

        concat_layer_sizes = [
            embedding_size * 2  # i changed it to *2 only because we dont have the loop_tensor_vector
        ] + self.comp_embed_layer_sizes[-2:]

        self.drops = [0.225, 0.225, 0.225, 0.225]

        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()

        for i in range(len(concat_layer_sizes) - 1):
            linear_concat = nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            initialization_function_xavier(linear_concat.weight)
            self.concat_layers.append(linear_concat)

            self.concat_dropouts.append(nn.Dropout(self.drops[i]))

        self.ELU = nn.ELU()

    def get_hidden_state(self, node):
        if node is not None and node.children != []:
            nodes_list = []

            for n in node.children:
                # Recusrive call to embed all the children of the loop first if they exist
                nodes_list.append(self.get_hidden_state(n))

            # Pass the embedding of all the child loops through the nodes LSTM
            nodes_tensor = torch.cat(nodes_list, 1)
            if not self.use_attention:
                lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
                nodes_h_n = nodes_h_n.permute(1, 0, 2)

        else:  # If there are no child loops contained within this level
            # The nodes embedding is a random vector (no_nodes_tensor) that represents that there are no nodes underneath this level
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                1, -1, -1
            )

        if node is not None and node.vector is not None:
            comps_tensor = torch.unsqueeze(torch.unsqueeze(torch.tensor(node.vector, dtype=torch.float32), dim=0), dim=0)
            if not self.use_attention:
                lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(comps_tensor)

        else:  # If there are no child computations contained within this level
            # The computations embedding is a random vector (no_comps_tensor) that represents that there are no computations underneath this level
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                1,  # i changed it to 1 for now
                -1,
                -1
            )

        # Concatinate the loop vector, computations embedding and nodes (child loops) embedding
        x = torch.cat((nodes_h_n, comps_h_n), 2)
        # Pass the concatinated vector through a feed forward neural network

        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))

        return x

    def get_hidden_state_batch(self, root_nodes):
        """
        root_nodes: list of LoopNode objects (length = batch_size)
        returns: tensor of shape (batch_size, 1, embedding_dim)
        """
        embeddings = []
        for node in root_nodes:
            emb = self.get_hidden_state(node)  # (1, 1, embedding_dim)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)  # (batch_size, 1, embedding_dim)

    def forward(self, obs):
        # batch_size = obs.size(0)

        consumer_obs = obs[:, :self.op_feats_size]
        producer_obs = obs[:, self.op_feats_size:(self.op_feats_size) * 2]

        rest = obs[:, (self.op_feats_size) * 2:-1]  # (B, N)

        num_loops = obs[:, -1]

        consumer_nodes = build_tree(consumer_obs, num_loops)
        producer_nodes = build_tree(producer_obs, num_loops)
        consumer_embeddings = self.get_hidden_state_batch(consumer_nodes)  # (B, 1, D)
        producer_embeddings = self.get_hidden_state_batch(producer_nodes)  # (B, 1, D)

        roots_tensor = torch.cat([consumer_embeddings, producer_embeddings], dim=1)  # (B, 2, D)
        _, (final_hidden, _) = self.lstm(roots_tensor)  # final_hidden: (1, B, D)

        final_hidden = final_hidden.squeeze(0)  # (B, D)

        out = torch.cat([final_hidden, rest], dim=1)  # (B, D + N)

        return out


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

    def __call__(self, obs: torch.Tensor, next_obs: torch.Tensor, actions_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        return super().__call__(obs, next_obs, actions_index)

    def forward(self, obs: torch.Tensor, next_obs: torch.Tensor, actions_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of the model.

        Args:
            obs (torch.Tensor): The input tensor.
            next_obs (torch.Tensor): The next input tensor.
            actions_index (torch.Tensor): The list of actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The logits of the transformations, parallelizations, tilings, and interchanges.
        """
        x = obs[:, :-(self.action_mask_size)]
        next_x = next_obs[:, :-(self.action_mask_size)]
        action_mask = obs[:, -(self.action_mask_size):].bool()

        state_latent = self.encoder(x)
        next_state_latent = self.encoder(next_x)

        next_state_latent_hat = self.forward_model(state_latent, actions_index)
        action_logits_hat = self.inverse_model(state_latent, next_state_latent, action_mask)

        return next_state_latent, next_state_latent_hat, action_logits_hat

    def loss(self, next_states_latent: torch.Tensor, next_states_latent_hat: torch.Tensor, action_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], actions_index: torch.Tensor) -> torch.Tensor:
        """Calculate the ICM loss.

        Args:
            next_states_latent (torch.Tensor): The next latent state tensor.
            next_states_latent_hat (torch.Tensor): The predicted next latent state tensor.
            action_logits (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The predicted logits of the actions.
            actions_index (list[tuple[str, Optional[Union[list[int], int]]]): The list of actions.

        Returns:
            torch.Tensor: The ICM loss.
        """
        return cfg.forward_weight * self.forward_model.loss(next_states_latent, next_states_latent_hat) + (1 - cfg.forward_weight) * self.inverse_model.loss(action_logits, actions_index)


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

    def __call__(self, state_latent: torch.Tensor, actions_index: torch.Tensor) -> torch.Tensor:
        return super().__call__(state_latent, actions_index)

    def forward(self, state_latent: torch.Tensor, actions_index: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            state_latent (torch.Tensor): The latent state tensor.
            actions_index (torch.Tensor): The list of actions.

        Returns:
            torch.Tensor: The predicted latent state tensor.
        """
        batch_size = state_latent.shape[0]

        transformation_index, parallelization_index, tiling_index, interchange_index = decode_actions_index(actions_index)

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

    def loss(self, action_logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], actions_index: torch.Tensor) -> torch.Tensor:
        """Calculate the inverse model loss.

        Args:
            action_logits (tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]): The predicted logits of the actions.
            actions_index (torch.Tensor): The list of actions.

        Returns:
            torch.Tensor: The inverse model loss.
        """
        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        batch_size = actions_index.size(0)

        transformation_index, parallelization_index, tiling_index, interchange_index = decode_actions_index(actions_index)
        transformation_logits_hat, parallelization_logits_hat, tiling_logits_hat, interchange_logits_hat = action_logits

        transformation_loss = self.disc_loss(transformation_logits_hat, transformation_index)
        parallelization_loss = self.disc_loss(parallelization_logits_hat.reshape(batch_size * L, TS + 1), parallelization_index.reshape(-1)).reshape(batch_size, L).sum(-1)
        tiling_loss = self.disc_loss(tiling_logits_hat.reshape(batch_size * L, TS + 1), tiling_index.reshape(-1)).reshape(batch_size, L).sum(-1)
        if cfg.interchange_mode == 'continuous':
            interchange_loss = self.cont_loss(interchange_logits_hat.squeeze(-1), interchange_index.float())
        else:
            interchange_loss = self.disc_loss(interchange_logits_hat, interchange_index)

        loss = transformation_loss
        loss[transformation_index == 1] += parallelization_loss[transformation_index == 1]
        loss[transformation_index == 2] += tiling_loss[transformation_index == 2]
        loss[transformation_index == 3] += interchange_loss[transformation_index == 3]

        return loss.mean()


def extract_masks(action_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract masks from the action mask tensor.

    Args:
        action_mask (torch.Tensor): The action mask tensor.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: The masks for the transformations, parallelizations, tilings, and interchanges.
    """
    batch_size = action_mask.shape[0]
    N = cfg.num_transformations
    L = cfg.max_num_loops
    TS = cfg.num_tile_sizes
    TP_BEGIN = N
    T_BEGIN = TP_BEGIN + L * (TS + 1)
    F_BEGIN = T_BEGIN + L * (TS + 1)
    I_BEGIN = F_BEGIN + L * (TS + 1)

    transform_mask = action_mask[:, :N]
    TP_mask = action_mask[:, TP_BEGIN:T_BEGIN].reshape(batch_size, L, TS + 1)
    T_mask = action_mask[:, T_BEGIN:F_BEGIN].reshape(batch_size, L, TS + 1)
    F_mask = action_mask[:, F_BEGIN:I_BEGIN].reshape(batch_size, L, TS + 1)
    if cfg.interchange_mode == 'continuous':
        I_mask = torch.ones((batch_size, 1), dtype=torch.bool)
    else:
        I_mask = action_mask[:, I_BEGIN:]

    return transform_mask, TP_mask, T_mask, I_mask, F_mask


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
        'fusion': 5
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
        5: 'fusion',
    }[transformation]


def decode_actions_index(index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert actions index to separate indices
    Args:
        torch.Tensor: the actions index
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The indices tensors for the transformations, parallelizations, tilings, and interchanges.
    """
    L = cfg.max_num_loops
    TP_BEGIN = 1
    T_BEGIN = TP_BEGIN + L
    F_BEGIN = T_BEGIN + L
    I_BEGIN = F_BEGIN + L

    transformation_index = index[:, 0]
    parallelization_index = index[:, TP_BEGIN:T_BEGIN]
    tiling_index = index[:, T_BEGIN:F_BEGIN]
    fusion_index = index[:, F_BEGIN:I_BEGIN]
    interchange_index = index[:, I_BEGIN]

    return transformation_index, parallelization_index, tiling_index, interchange_index, fusion_index


def encode_actions_index(transformation_index: torch.Tensor, parallelization_index: torch.Tensor, tiling_index: torch.Tensor, interchange_index: torch.Tensor, fusion_index: torch.Tensor) -> torch.Tensor:
    """Convert separate indices to actions index
    Args:
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor: The indices tensors for the transformations, parallelizations, tilings, and interchanges.
    Returns:
        torch.Tensor: the action index
    """
    return torch.cat((
        transformation_index.unsqueeze(-1),
        parallelization_index,
        tiling_index,
        fusion_index,
        interchange_index.unsqueeze(-1),
    ), dim=-1)


def raw_actions_to_indices(actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    fusion_index = torch.zeros((batch_size, L), dtype=torch.int64)
    for i, action in enumerate(actions):
        action_name, parameters = action
        match action_name:
            case 'parallelization':
                parallelization_index[i, :len(parameters)] = torch.tensor(parameters)
            case 'tiling':
                tiling_index[i, :len(parameters)] = torch.tensor(parameters)
            case 'interchange':
                interchange_index[i] = parameters
            case 'fusion':
                fusion_index[i, :len(parameters)] = torch.tensor(parameters)

    return transformation_index, parallelization_index, tiling_index, interchange_index, fusion_index


def indices_to_raw_actions(transformation_index: torch.Tensor, parallelization_index: torch.Tensor, tiling_index: torch.Tensor, interchange_index: torch.Tensor, fusion_index: torch.Tensor, num_loops: torch.Tensor) -> list[tuple[str, Optional[Union[list[int], int]]]]:
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
            case 'fusion':
                parameters = fusion_index[i, :num_loops[i]].tolist()
        actions.append((transformation, parameters))

    return actions
