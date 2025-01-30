import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Optional, Union
from rl_autoschedular import config as cfg


class HiearchyModel(nn.Module):
    """Hierarchical reinforcement learning model for MLIR code optimization."""
    def __init__(self):
        """Initialize the model."""
        super(HiearchyModel, self).__init__()

        L = cfg.max_num_loops
        D = cfg.max_num_load_store_dim
        SD = cfg.max_num_stores_loads
        self.input_dim = 1 + L + L * D * SD + L * D + 5 + 1 + L * 3 * cfg.truncate
        self.num_loops = L
        self.num_transformations = cfg.num_transformations
        self.num_tiles = cfg.num_tile_sizes

        self.action_mask_size = self.num_transformations + 3 * self.num_loops

        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
        )

        self.value_network = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.transformation_selection = nn.Linear(512, self.num_transformations)  # +1 for the stop operation
        self.interchange_fc = nn.Linear(512, self.num_loops)
        self.tiling_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no tiling
        self.parall_fc = nn.Linear(512, self.num_loops * (self.num_tiles + 1))  # +1 for the no parallelizattion

    def sample(self, obs: torch.Tensor, actions: Optional[list[tuple[str, Optional[Union[list[int], int]]]]] = None) -> tuple[list[tuple[str, Optional[Union[list[int], int]]]], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action from the model.

        Args:
            obs (torch.Tensor): The input tensor.
            actions (Optional[list[tuple[str, Optional[Union[list[int], int]]]]]): list of actions forced for the model to return. Defaults to None.

        Returns:
            Optional[list[tuple[str, Optional[Union[list[int], int]]]]]: list of actions.
            torch.Tensor: action log probabilities.
            torch.Tensor: action values.
            torch.Tensor: resulting entropy.
        """

        *leading_dims, _ = obs.shape

        # Spint `obs` into the input `x` and the `action_mask`
        x = obs[..., :-(self.action_mask_size)]
        action_mask = obs[..., -(self.action_mask_size):].bool()

        # print(action_mask)

        # decompose action mask:
        L = self.num_loops

        TP_BEGIN = self.num_transformations
        T_BEGIN = TP_BEGIN + L
        I_BEGIN = T_BEGIN + L

        # Define the mask of each transformation
        transform_mask = action_mask[..., :self.num_transformations]
        TP_mask = action_mask[..., TP_BEGIN:T_BEGIN]
        T_mask = action_mask[..., T_BEGIN:I_BEGIN]
        I_mask = action_mask[..., I_BEGIN:]

        # Model inference:
        x1 = self.backbone(x)
        transformation_logits = self.transformation_selection(x1)
        interchange_logits = self.interchange_fc(x1)
        tiling_logits = self.tiling_fc(x1)
        parall_logits = self.parall_fc(x1)

        values = self.value_network(x)

        tiling_logits = tiling_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)
        parall_logits = parall_logits.reshape(*leading_dims, self.num_loops, self.num_tiles + 1)

        # print(parall_logits.shape, tiling_logits.shape, interchange_logits.shape)

        # Apply the mask on the transformations:
        transformation_logits = torch.where(transform_mask, transformation_logits, -float('inf'))
        interchange_logits = torch.where(I_mask, interchange_logits, -float('inf'))

        # Get the actions indices:
        transformation_dist = Categorical(logits=transformation_logits)
        interchange_dist = Categorical(logits=interchange_logits)
        tiling_dist = Categorical(logits=tiling_logits)
        parall_dist = Categorical(logits=parall_logits)

        if actions is None:
            transformation_index = transformation_dist.sample()
            interchange_index = interchange_dist.sample()
            tiling_index = tiling_dist.sample()
            parall_index = parall_dist.sample()

        else:

            transformation_index = torch.zeros((len(actions),), dtype=torch.int64)
            parall_index = torch.zeros((len(actions), L), dtype=torch.int64)
            tiling_index = torch.zeros((len(actions), L), dtype=torch.int64)
            interchange_index = torch.zeros((len(actions),), dtype=torch.int64)

            for i, action in enumerate(actions):
                action_name, parameters = action
                if action_name == 'no_transformation':
                    transformation_index[i] = 0
                elif action_name == 'parallelization':
                    transformation_index[i] = 1
                    parall_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                elif action_name == 'tiling':
                    transformation_index[i] = 2
                    tiling_index[i] = torch.tensor(list(parameters) + [0] * (L - len(parameters)))
                elif action_name == 'interchange':
                    transformation_index[i] = 3
                    interchange_index[i] = parameters
                elif action_name == 'vectorization':
                    transformation_index[i] = 4
                elif action_name == 'img2col':
                    transformation_index[i] = 5

        # Get the action prob and log_prob
        transformation_log_p = F.log_softmax(transformation_logits, dim=-1).gather(-1, transformation_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        interchange_log_p = F.log_softmax(interchange_logits, dim=-1).gather(-1, interchange_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        tiling_log_p = F.log_softmax(tiling_logits, dim=-1).gather(-1, tiling_index.unsqueeze(-1)).reshape(*leading_dims, -1)
        parall_log_p = F.log_softmax(parall_logits, dim=-1).gather(-1, parall_index.unsqueeze(-1)).reshape(*leading_dims, -1)

        parall_log_p = torch.where(TP_mask, parall_log_p, 0).sum(-1, keepdim=True)
        tiling_log_p = torch.where(T_mask, tiling_log_p, 0).sum(-1, keepdim=True)

        actions = []
        for i in range(transformation_index.shape[0]):
            if transformation_index[i] == 0:
                actions.append(['no_transformation', None])

            elif transformation_index[i] == 1:
                params = []
                for j in range(parall_index[i].shape[0]):
                    if TP_mask[i, j]:
                        params.append(parall_index[i, j].item())
                actions.append(['parallelization', params])

            elif transformation_index[i] == 2:
                params = []
                for j in range(tiling_index[i].shape[0]):
                    if T_mask[i, j]:
                        params.append(tiling_index[i, j].item())
                actions.append(['tiling', params])

            elif transformation_index[i] == 3:
                actions.append(['interchange', interchange_index[i].item()])

            elif transformation_index[i] == 4:
                actions.append(['vectorization', None])

            elif transformation_index[i] == 5:
                actions.append(['img2col', None])

        transformation_log_p, interchange_log_p, tiling_log_p, parall_log_p = transformation_log_p.reshape(-1), interchange_log_p.reshape(-1), tiling_log_p.reshape(-1), parall_log_p.reshape(-1)

        is_no_action = (transformation_index == 0)
        is_parall = (transformation_index == 1)
        is_tiling = (transformation_index == 2)
        is_interchange = (transformation_index == 3)

        action_log_p = torch.zeros_like(transformation_index, dtype=torch.float32)
        action_log_p[is_interchange] = interchange_log_p[is_interchange] + transformation_log_p[is_interchange]
        action_log_p[is_tiling] = tiling_log_p[is_tiling] + transformation_log_p[is_tiling]
        action_log_p[is_parall] = parall_log_p[is_parall] + transformation_log_p[is_parall]
        action_log_p[is_no_action] = transformation_log_p[is_no_action]

        entropy = transformation_dist.entropy().mean() + interchange_dist.entropy().mean() + tiling_dist.entropy().mean() + parall_dist.entropy().mean()

        return actions, action_log_p, values, entropy
        # return action_log_p, entropy, values, sub_entropies
