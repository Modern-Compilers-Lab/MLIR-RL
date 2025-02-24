import torch
import torch.nn as nn
from torch.distributions import Categorical, Binomial
from typing import Optional, Union
from rl_autoschedular import config as cfg
from rl_autoschedular.truncated_normal import TruncatedNormal


class HiearchyModel(nn.Module):
    """Hierarchical reinforcement learning model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(HiearchyModel, self).__init__()

        N = cfg.num_transformations
        L = cfg.max_num_loops
        D = cfg.max_num_load_store_dim
        SD = cfg.max_num_stores_loads
        T = cfg.num_tile_sizes

        match cfg.interchange_mode:
            case 'enumerate':
                interchange_mask = 3 * L - 6
                interchange_input = 0
                interchange_layer = nn.Linear(512, 3 * L - 6)
            case 'pointers':
                interchange_mask = L
                interchange_input = L
                interchange_layer = nn.Linear(512, L)
            case 'continuous':
                interchange_mask = 0
                interchange_input = 0
                interchange_layer = nn.Linear(512, 1)
                if cfg.interchange_distribution == 'normal':
                    self.interchange_logstd = nn.Parameter(torch.zeros(1))

        self.input_dim = 5 + L + L * D * SD + L * D + 5 + interchange_input + cfg.truncate * 3 * L
        self.action_mask_size = N + L + L + interchange_mask
        activation_layer = nn.ReLU if cfg.activation == 'relu' else nn.Tanh

        if cfg.new_architecture:
            self.backbone = nn.Sequential(
                nn.Linear(self.input_dim, 512),
                activation_layer(),
                nn.Linear(512, 512),
                activation_layer(),
            )

            self.value_network = nn.Sequential(
                nn.Linear(self.input_dim, 512),
                activation_layer(),
                nn.Linear(512, 512),
                activation_layer(),
                nn.Linear(512, 1),
            )

            self.transformation_selection = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                nn.Linear(512, N),
            )

            self.interchange_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                interchange_layer,
            )

            self.tiling_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                nn.Linear(512, L * (T + 1)),
            )

            self.parallelization_fc = nn.Sequential(
                nn.Linear(512, 512),
                activation_layer(),
                nn.Linear(512, L * (T + 1)),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(self.input_dim, 512),
                activation_layer(),
                nn.Linear(512, 512),
                activation_layer(),
                nn.Linear(512, 512),
                activation_layer(),
            )

            self.value_network = nn.Sequential(
                nn.Linear(self.input_dim, 512),
                activation_layer(),
                nn.Linear(512, 512),
                activation_layer(),
                nn.Linear(512, 512),
                activation_layer(),
                nn.Linear(512, 1),
            )

            self.transformation_selection = nn.Linear(512, N)

            self.interchange_fc = interchange_layer

            self.tiling_fc = nn.Linear(512, L * (T + 1))

            self.parallelization_fc = nn.Linear(512, L * (T + 1))

    def sample(self, obs: torch.Tensor, num_loops: list[int], actions: Optional[list[tuple[str, Optional[Union[list[int], int]]]]] = None) -> tuple[list[tuple[str, Optional[Union[list[int], int]]]], torch.Tensor, torch.Tensor, torch.Tensor]:
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

        batch_size, _ = obs.shape
        assert actions is None or len(actions) == batch_size

        # Split `obs` into the input `x` and the `action_mask`
        x = obs[:, :-(self.action_mask_size)]
        action_mask = obs[:, -(self.action_mask_size):].bool()

        L = cfg.max_num_loops
        TS = cfg.num_tile_sizes
        TP_BEGIN = cfg.num_transformations
        T_BEGIN = TP_BEGIN + L
        I_BEGIN = T_BEGIN + L

        # Define the mask of each transformation
        transform_mask = action_mask[:, :cfg.num_transformations]
        TP_mask = action_mask[:, TP_BEGIN:T_BEGIN].unsqueeze(-1).broadcast_to(batch_size, L, TS + 1)
        TP_mask[:, :, 0] = 1  # Never mask "no tiling"
        T_mask = action_mask[:, T_BEGIN:I_BEGIN].unsqueeze(-1).broadcast_to(batch_size, L, TS + 1)
        T_mask[:, :, 0] = 1  # Never mask "no tiling"
        if cfg.interchange_mode == 'continuous':
            I_mask = torch.ones((batch_size, 1), dtype=torch.bool)
        else:
            I_mask = action_mask[:, I_BEGIN:]

        # Model inference:
        x1 = self.backbone(x)

        transformation_logits = self.transformation_selection(x1)
        parallelization_logits = self.parallelization_fc(x1).reshape(batch_size, L, TS + 1)
        tiling_logits = self.tiling_fc(x1).reshape(batch_size, L, TS + 1)
        interchange_logits = self.interchange_fc(x1)

        values = self.value_network(x)

        # Apply masks on logits
        transformation_logits = transformation_logits.where(transform_mask, -torch.inf)
        parallelization_logits = parallelization_logits.where(TP_mask, -torch.inf)
        tiling_logits = tiling_logits.where(T_mask, -torch.inf)
        interchange_logits = interchange_logits.where(I_mask, -torch.inf)

        # Create distributions with the masked probabilities
        transformation_dist = Categorical(logits=transformation_logits)
        parallelization_dist = Categorical(logits=parallelization_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        if cfg.interchange_mode == 'continuous':
            interchange_prob = interchange_logits.squeeze(-1).sigmoid()
            total_count = (torch.tensor(num_loops) + 1).lgamma().exp().long() - 1
            if cfg.interchange_distribution == 'binomial':
                interchange_dist = Binomial(total_count, probs=interchange_prob)
            else:
                interchange_dist = TruncatedNormal(
                    loc=interchange_prob * total_count,
                    scale=self.interchange_logstd.exp(),
                    a=0,
                    b=torch.maximum(total_count, torch.tensor(1)),
                )
        else:
            interchange_dist = Categorical(logits=interchange_logits)

        # Get chosen actions and their indices
        with torch.no_grad():
            if actions is None:
                # Sample actions
                transformation_index = transformation_dist.sample()
                parallelization_index = parallelization_dist.sample()
                tiling_index = tiling_dist.sample()
                if cfg.interchange_mode == 'continuous' and cfg.use_interchange_mean:
                    interchange_index = interchange_dist.mean.long()
                else:
                    interchange_index = interchange_dist.sample().long()

                # Get raw actions from indices
                actions = self.__indices_to_raw_actions(transformation_index, parallelization_index, tiling_index, interchange_index, num_loops)
            else:
                transformation_index, parallelization_index, tiling_index, interchange_index = self.__raw_actions_to_indices(actions)

        # Get log probabilities of the actions
        transformation_log_p = transformation_dist.log_prob(transformation_index)
        interchange_log_p = interchange_dist.log_prob(interchange_index)
        parallelization_log_p = parallelization_dist.log_prob(parallelization_index).sum(-1)
        tiling_log_p = tiling_dist.log_prob(tiling_index).sum(-1)

        # Calculate the total log probability
        action_log_p = transformation_log_p
        action_log_p += parallelization_log_p * (transformation_index == 1)
        action_log_p += tiling_log_p * (transformation_index == 2)
        action_log_p += interchange_log_p * (transformation_index == 3)

        entropy = transformation_dist.entropy().mean()
        entropy += (parallelization_dist.entropy().sum(-1) * (transformation_index == 1)).mean()
        entropy += (tiling_dist.entropy().sum(-1) * (transformation_index == 2)).mean()
        if not isinstance(interchange_dist, Binomial) or batch_size == 1:
            entropy += (interchange_dist.entropy() * (transformation_index == 3)).mean()

        return actions, action_log_p, values, entropy

    def sample_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Sample the value from the model.

        Args:
            obs (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The value tensor.
        """
        return self.value_network(obs[:, :-(self.action_mask_size)])

    def __transformation_to_int(self, transformation: str) -> int:
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
            'img2col': 5,
        }[transformation]

    def __int_to_transformation(self, transformation: int) -> str:
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
            5: 'img2col',
        }[transformation]

    def __raw_actions_to_indices(self, actions: list[tuple[str, Optional[Union[list[int], int]]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a list of actions to tensor of indices.

        Args:
            actions (Optional[list[tuple[str, Optional[Union[list[int], int]]]]): The list of actions.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The indices tensors for the transformations, parallelizations, tilings, and interchanges.
        """
        L = cfg.max_num_loops
        batch_size = len(actions)
        transformation_index = torch.tensor([self.__transformation_to_int(name) for name, _ in actions], dtype=torch.int64)

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

    def __indices_to_raw_actions(self, transformation_index: torch.Tensor, parallelization_index: torch.Tensor, tiling_index: torch.Tensor, interchange_index: torch.Tensor, num_loops: list[int]) -> list[tuple[str, Optional[Union[list[int], int]]]]:
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
            transformation = self.__int_to_transformation(transformation_index[i].item())
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
