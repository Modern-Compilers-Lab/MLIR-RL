from utils.config import Config
from .base import Action
from .no_transformation import NoTransformation
from .tiling import Tiling
from .tiled_parallelization import TiledParallelization
from .tiled_fusion import TiledFusion
from .interchange import Interchange
from .vectorization import Vectorization
from rl_autoschedular.state import OperationState
import torch
from torch.distributions import Distribution, Categorical
from typing import Optional


class ActionSpace:
    """Class holding information about the action space"""

    supported_actions: list[type[Action]] = [
        NoTransformation,
        Tiling,
        TiledParallelization,
        TiledFusion,
        Interchange,
        Vectorization
    ]

    @classmethod
    def size(cls):
        return len(cls.supported_actions)

    @classmethod
    def cumulative_params_sizes(cls):
        sizes: list[int] = [1]
        for trans in cls.supported_actions:
            sizes.append(sizes[-1] + trans.params_size())
        return sizes

    @classmethod
    def cumulative_mask_sizes(cls):
        sizes: list[int] = [cls.size()]
        for trans in cls.supported_actions:
            sizes.append(sizes[-1] + trans.mask_size())
        return sizes

    @classmethod
    def cumulative_history_sizes(cls):
        sizes: list[int] = [0]
        for trans in cls.supported_actions:
            sizes.append(sizes[-1] + trans.history_size())
        return sizes

    @classmethod
    def action_by_index(cls, index: torch.Tensor, state: OperationState) -> Action:
        action_idx = int(index[0].item())
        action_type = cls.supported_actions[action_idx]
        if not action_type.params_size():
            return action_type(state)

        cum_sizes = cls.cumulative_params_sizes()
        params = index[cum_sizes[action_idx]:cum_sizes[action_idx + 1]].long().tolist()
        return action_type(params, state)

    @classmethod
    def action_number(cls, action_type: type[Action]) -> int:
        return cls.supported_actions.index(action_type)

    @classmethod
    def action_type_by_symbol(cls, symbol: str) -> type[Action]:
        for action in cls.supported_actions:
            if action.symbol == symbol:
                return action

        raise ValueError(f"action symbol '{symbol}' not supported")

    @classmethod
    def action_number_by_symbol(cls, symbol: str) -> int:
        return cls.action_number(cls.action_type_by_symbol(symbol))

    @classmethod
    def action_mask(cls, state: OperationState) -> torch.Tensor:
        cfg = Config()
        mask = torch.zeros(cls.size(), dtype=torch.bool)

        def allow_action(a: type[Action]):
            if a.is_allowed(state):
                mask[cls.action_number(a)] = True

        def forbid_action(a: type[Action]):
            mask[cls.action_number(a)] = False

        def allow_all():
            for action in cls.supported_actions:
                allow_action(action)

        # If state is terminal don't allow any further actions
        if not state.terminal:
            if Interchange.incomplete_interchange(state):
                # Special case where interchange isn't complete yet
                mask[cls.action_number(Interchange)] = True
            elif state.has_incomplete_action:
                raise Exception(
                    f"Unhandled incomplete action {state.latest_action}"
                    ", can't move on to another step"
                )
            elif cfg.order:
                # Enforce order if provided
                if state.step_count >= len(cfg.order):
                    raise Exception("actions order must be ended with a terminal action")
                if not cfg.order[state.step_count]:
                    # If at current step nothing is specified, allow everything
                    allow_all()
                elif cfg.order[state.step_count][0] == '!':
                    # Forbid actions
                    allow_all()
                    for s in cfg.order[state.step_count][1:]:
                        forbid_action(cls.action_type_by_symbol(s))
                else:
                    # Allow actions
                    for s in cfg.order[state.step_count]:
                        allow_action(cls.action_type_by_symbol(s))
            else:
                # If none of the above applies, allow everything
                allow_all()

            # Check that there is at least one action allowed
            if not mask.any():
                raise Exception(f"no actions allowed for the current state at step {state.step_count}")

        for action in cls.supported_actions:
            action_mask = action.action_mask(state)
            if action_mask is None:
                continue

            mask = torch.cat((mask, action_mask))

        return mask

    @classmethod
    def action_history(cls, seq: list[Action]) -> torch.Tensor:
        history = []
        for action in cls.supported_actions:
            action_history = action.action_history(seq)
            if action_history is None:
                continue
            history.append(action_history)
        if not history:
            return torch.tensor([])

        return torch.cat(history)

    @classmethod
    def distributions(cls, obs: torch.Tensor, selection_logits: torch.Tensor, *actions_logits: Optional[torch.Tensor]) -> list[Optional[Distribution]]:
        """Create a list of distributions for the actions based on the logits.

        Args:
            obs (torch.Tensor): Observation tensor.
            selection_logits (torch.Tensor): Logits for action selection.
            *actions_logits (torch.Tensor): Logits for each action's parameters.

        Returns:
            list[Distribution]: List of distributions for each action.
        """
        from rl_autoschedular.observation import Observation, ActionMask

        actions_mask = Observation.get_part(obs, ActionMask).bool()
        dists_list: list[Optional[Distribution]] = [
            Categorical(logits=selection_logits.where(actions_mask[:, :cls.size()], -torch.inf))
        ]
        cum_sizes = cls.cumulative_mask_sizes()
        for i, action in enumerate(cls.supported_actions):
            if not action.mask_size():
                dists_list.append(None)
                continue

            assert actions_logits[i] is not None, f"action '{action.symbol}' must have logits"
            masked_logits = actions_logits[i].where(actions_mask[:, cum_sizes[i]:cum_sizes[i + 1]], -torch.inf)
            dists_list.append(action.distribution(masked_logits))

        return dists_list

    @classmethod
    def uniform_distributions(cls, obs: torch.Tensor) -> list[Optional[Distribution]]:
        """Create a list of uniform distributions for the actions based on the observation.

        Args:
            obs (torch.Tensor): Observation tensor.

        Returns:
            list[Distribution]: List of distributions for each action.
        """
        from rl_autoschedular.observation import Observation, ActionMask, NumLoops

        actions_mask = Observation.get_part(obs, ActionMask).bool()
        num_loops = Observation.get_part(obs, NumLoops)
        selection_mask = actions_mask[:, :cls.size()]
        dists_list: list[Optional[Distribution]] = [
            Categorical(logits=torch.zeros_like(selection_mask).where(selection_mask, -torch.inf))
        ]
        cum_sizes = cls.cumulative_mask_sizes()
        for i, action in enumerate(cls.supported_actions):
            if not action.mask_size():
                dists_list.append(None)
                continue

            action_mask = actions_mask[:, cum_sizes[i]:cum_sizes[i + 1]]
            logits = torch.zeros_like(action_mask).where(action_mask, -torch.inf)
            dists_list.append(action.uniform_distribution(logits, num_loops))

        return dists_list

    @classmethod
    def distributions_stats(cls, distributions: list[Optional[Distribution]], index: torch.Tensor, eps_distributions: Optional[list[Optional[Distribution]]] = None, eps: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        eps_distributions: list[Optional[Distribution]] = eps_distributions or [None] * len(distributions)

        selection_index = index[:, 0]
        selection_dist = distributions[0]
        selection_eps_dist = eps_distributions[0]
        selection_log_p = selection_dist.log_prob(selection_index)
        if eps is not None:
            selection_eps_log_p = selection_eps_dist.log_prob(selection_index)
            selection_log_p = (selection_log_p.exp() * (1 - eps) + selection_eps_log_p.exp() * eps).log()

        cum_sizes = cls.cumulative_params_sizes()
        actions_log_p, entropies = selection_log_p, selection_dist.entropy()
        for i, (action, dist, eps_dist) in enumerate(zip(cls.supported_actions, distributions[1:], eps_distributions[1:])):
            if dist is None:
                continue

            action_index = index[:, cum_sizes[i]:cum_sizes[i + 1]]
            action_log_p, entropy = action.distribution_stats(dist, action_index, eps_dist, eps)
            actions_log_p[selection_index == i] += action_log_p[selection_index == i]
            entropies[selection_index == i] += entropy[selection_index == i]

        return actions_log_p, entropies

    @classmethod
    def sample(cls, obs: torch.Tensor, distributions: list[Optional[Distribution]], eps_distributions: list[Optional[Distribution]], uniform: bool = False, greedy: bool = False) -> torch.Tensor:
        assert not uniform or not greedy, "can't sample uniformly and greedily at once"
        from rl_autoschedular.observation import Observation, NumLoops

        num_loops = Observation.get_part(obs, NumLoops)
        selection_dist = distributions[0]
        selection_eps_dist = eps_distributions[0]

        if greedy:
            selection_index = selection_dist.probs.argmax(-1)
        elif uniform:
            selection_index = selection_eps_dist.sample()
        else:
            selection_index = selection_dist.sample()

        index = selection_index.unsqueeze(-1)
        for action, dist, eps_dist in zip(cls.supported_actions, distributions[1:], eps_distributions[1:]):
            if dist is None:
                continue

            index = torch.cat((index, action.sample(dist, eps_dist, num_loops, uniform, greedy)), dim=1)

        return index
