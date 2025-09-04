from rl_autoschedular.state import OperationState
from rl_autoschedular.transforms import transform_tile
from typing import Optional

from utils.config import Config
from .base import Action
import torch
import math
from torch.distributions import Categorical


class Tiling(Action):
    """Class representing Tiling action"""

    symbol = 'T'

    parameters: list[int]

    def __init__(self, parameters: list[int], state: Optional[OperationState] = None, **extras):
        if state:
            # Case where state is provided -> Parameters need processing

            tile_sizes = []
            for param, loop in zip(parameters, state.operation_features.nested_loops):
                if param == 0:
                    tile_sizes.append(0)
                else:
                    ts = 2 ** (param - 1)
                    assert loop.upper_bound % ts == 0 and loop.upper_bound != ts, \
                        f'Tiling parameter {param} is not a factor of loop upper bound {loop.upper_bound}'
                    tile_sizes.append(ts)
            parameters = tile_sizes
        super().__init__(parameters, state, **extras)

    @classmethod
    def params_size(cls):
        return Config().max_num_loops

    @classmethod
    def network_output_size(cls):
        return Config().max_num_loops * (Config().num_tile_sizes + 1)

    @classmethod
    def history_size(cls):
        return Config().truncate * Config().max_num_loops * (Config().num_tile_sizes + 1)

    @classmethod
    def action_mask(cls, state: OperationState):
        mask = torch.zeros((Config().max_num_loops, Config().num_tile_sizes + 1), dtype=torch.bool)
        mask[:, 0] = True
        for i, loop in enumerate(state.operation_features.nested_loops):
            ts_count = cls.__get_tiles_count(loop.upper_bound)
            mask[i, :ts_count] = True

        return mask.reshape(-1)

    @classmethod
    def action_history(cls, seq):
        history = torch.zeros((Config().truncate, Config().max_num_loops, Config().num_tile_sizes + 1))
        for i, action in enumerate(seq):
            if not isinstance(action, Tiling):
                continue

            for j, param in enumerate(action.parameters):
                if param == 0:
                    history[i, j, 0] = 1
                else:
                    assert param > 0 and (param & (param - 1) == 0), f'Expected tile size to be a positive power of 2, found {param}'
                    ts_index = int(math.log2(param)) + 1
                    assert ts_index < history.size(2), f'Overflow of tile size, max size is {2 ** (Config().num_tile_sizes - 1)} found {param}'
                    history[i, j, ts_index] = 1

        return history.reshape(-1)

    @classmethod
    def distribution(cls, logits):
        logits = logits.reshape(-1, Config().max_num_loops, Config().num_tile_sizes + 1)
        return Categorical(logits=logits)

    @classmethod
    def distribution_stats(cls, distribution, index, eps_distribution, eps=None):
        log_p = distribution.log_prob(index).sum(-1)

        if eps is not None:
            eps_log_p = eps_distribution.log_prob(index).sum(-1)
            log_p = (log_p.exp() * (1 - eps) + eps_log_p.exp() * eps).log()

        entropy = distribution.entropy().sum(-1)

        return log_p, entropy

    @classmethod
    def sample(cls, distribution, eps_distribution, num_loops, uniform, greedy):
        if greedy:
            index = distribution.probs.argmax(-1)
        elif uniform:
            index = eps_distribution.sample()
        else:
            index = distribution.sample()

        return index

    def _apply_ready(self, code):
        return transform_tile(code, self.operation_tag, self.parameters)

    def update_features(self, operation_features):
        # A tiled operation loses its producers outside the tiling loop
        operation_features.producers = []

        new_operation_features = operation_features.copy()
        for nested_loop, tile_size in zip(new_operation_features.nested_loops, self.parameters):
            if tile_size == 0:
                continue
            nested_loop.upper_bound = tile_size

        return new_operation_features

    @staticmethod
    def __get_tiles_count(ub: int) -> int:
        """Get the number of tiling candidates for a given loop upper bound.

        Args:
            ub (int): The loop upper bound.

        Returns:
            int: The number of candidates.
        """
        for i in range(Config().num_tile_sizes):
            ts = 2 ** i
            if ub % ts != 0 or ub == ts:
                return i + 1
        return Config().num_tile_sizes + 1
