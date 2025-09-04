from utils.config import Config
from .base import Action
from rl_autoschedular.state import OperationState, OperationType
from rl_autoschedular.transforms import transform_interchange
from typing import Optional
from enum import Enum
import torch
from torch.distributions import Categorical, Normal, Uniform
import math


class InterchangeMethod(Enum):
    EnumeratedCandidates = 'enumerate'
    LevelsPointers = 'pointers'
    ContinuousEncoding = 'continuous'


class Interchange(Action):
    """Class representing Interchange action"""

    symbol = 'I'

    parameters: list[int]

    # --- constants ---
    method = InterchangeMethod(Config().interchange_mode)
    log_std = torch.nn.Parameter(torch.zeros(1))

    def __init__(self, parameters: list[int], state: Optional[OperationState] = None, **extras):
        if state:
            # Case where state is provided -> Parameters need processing

            assert len(parameters) == 1, 'uncompatible parameters for constructor call'
            parameter = parameters[0]
            num_loops = len(state.operation_features.nested_loops)
            match Interchange.method:
                case InterchangeMethod.EnumeratedCandidates:
                    parameters = self.__get_candidates(num_loops)[parameter]
                case InterchangeMethod.ContinuousEncoding:
                    parameters = self.__decode_continuous(parameter, num_loops)
                case InterchangeMethod.LevelsPointers:
                    old_action = self.incomplete_interchange(state)
                    if old_action:
                        perm_buffer = old_action.parameters
                    else:
                        perm_buffer = []

                    assert parameter not in perm_buffer, 'repitition detected in permutation'
                    parameters = perm_buffer + [parameter]
                    assert len(parameters) <= num_loops, 'interchange parameter exceeds number of loops'
                    if len(parameters) < num_loops:
                        self.ready = False
        super().__init__(parameters, state, **extras)

    @classmethod
    def params_size(cls):
        return 1

    @classmethod
    def network_output_size(cls):
        match cls.method:
            case InterchangeMethod.EnumeratedCandidates:
                return 3 * Config().max_num_loops - 6
            case InterchangeMethod.LevelsPointers:
                return Config().max_num_loops
            case InterchangeMethod.ContinuousEncoding:
                return 1

    @classmethod
    def history_size(cls):
        return Config().truncate * Config().max_num_loops * Config().max_num_loops

    @classmethod
    def action_mask(cls, state):
        L = Config().max_num_loops
        I_BEGIN_2C = L - 1
        I_BEGIN_3C = I_BEGIN_2C + L - 2

        num_loops = len(state.operation_features.nested_loops)
        mask = torch.ones(cls.mask_size(), dtype=torch.bool)
        match cls.method:
            case InterchangeMethod.ContinuousEncoding:
                pass
            case InterchangeMethod.EnumeratedCandidates:
                if num_loops == 1:
                    mask[1:] = False
                else:
                    mask[num_loops - 1:I_BEGIN_2C] = False
                    mask[I_BEGIN_2C + num_loops - 2:I_BEGIN_3C] = False
                    mask[I_BEGIN_3C + max(num_loops - 3, 0):] = False
            case InterchangeMethod.LevelsPointers:
                mask[num_loops:] = False
                old_action = cls.incomplete_interchange(state)
                if old_action:
                    for param in old_action.parameters:
                        mask[param] = False

        return mask

    @classmethod
    def action_history(cls, seq):
        history = torch.zeros((Config().truncate, Config().max_num_loops, Config().max_num_loops))
        for i, action in enumerate(seq):
            if not isinstance(action, Interchange):
                continue

            for j, param in enumerate(action.parameters):
                history[i, j, param] = 1

        return history.reshape(-1)

    @classmethod
    def distribution(cls, logits):
        match cls.method:
            case InterchangeMethod.EnumeratedCandidates | InterchangeMethod.LevelsPointers:
                return Categorical(logits=logits)
            case InterchangeMethod.ContinuousEncoding:
                logit = logits.squeeze(-1)
                return Normal(logit, cls.log_std.clamp(-1, 1).exp())

    @classmethod
    def uniform_distribution(cls, logits, num_loops):
        match cls.method:
            case InterchangeMethod.EnumeratedCandidates | InterchangeMethod.LevelsPointers:
                return Categorical(logits=logits)
            case InterchangeMethod.ContinuousEncoding:
                total_count = (num_loops + 1).lgamma().exp()
                return Uniform(0.0, total_count)

    @classmethod
    def distribution_stats(cls, distribution, index, eps_distribution, eps=None):
        index = index.squeeze(-1)
        if isinstance(distribution, Normal):
            # Special case in Normal distribution we need to consider all
            # the interval [i,i+1), so we use log CDF instead of log P
            log_p = (distribution.cdf(index + 1) - distribution.cdf(index) + 1e-8).log()
        else:
            log_p = distribution.log_prob(index)

        if eps is not None:
            eps_log_p = eps_distribution.log_prob(index)
            log_p = (log_p.exp() * (1 - eps) + eps_log_p.exp() * eps).log()

        entropy = distribution.entropy()

        return log_p, entropy

    @classmethod
    def sample(cls, distribution, eps_distribution, num_loops, uniform, greedy):
        if greedy:
            if cls.method == InterchangeMethod.ContinuousEncoding:
                index = distribution.mean.long()
            else:
                index = distribution.probs.argmax(-1)
        elif uniform:
            index = eps_distribution.sample().long()
        else:
            index = distribution.sample().long()

        if cls.method == InterchangeMethod.ContinuousEncoding:
            total_count = (num_loops + 1).lgamma().exp().long()
            index = index.clamp(torch.zeros_like(total_count).long(), total_count - 1)

        return index.unsqueeze(-1)

    def _apply_ready(self, code):
        return transform_interchange(code, self.operation_tag, self.parameters)

    def update_features(self, operation_features):
        if not self.ready:
            return operation_features

        new_operation_features = operation_features.copy()
        for i, j in enumerate(self.parameters):
            new_operation_features.nested_loops[i] = operation_features.nested_loops[j]

        # In case an interchange was applied to pooling or conv, vectorization is no longer possible
        if operation_features.operation_type in [OperationType.Pooling, OperationType.Conv] and self.parameters != list(range(len(self.parameters))):
            new_operation_features.vectorizable = False

        return new_operation_features

    @staticmethod
    def __decode_continuous(parameter: int, num_loops: int) -> list[int]:
        """Decode the interchange parameter to get the loop permutation.

        Args:
            parameter (int): The interchange parameter.
            num_loops (int): The number of loops in the operation.

        Returns:
            list[int]: The loop permutation.
        """
        x = parameter
        n = num_loops
        if x >= math.factorial(n):
            raise Exception(f"Invalid interchange parameter: {x}")

        # Convert x to factorial number
        fact_x = '0'
        q = x
        d = 2
        while q > 0:
            r = q % d
            q = q // d
            fact_x = str(r) + fact_x
            d += 1

        # Ensure to get exactly n digits
        fact_x = fact_x.zfill(n)[-n:]

        # Decode factorial number following Lehmer code
        nl = list(map(int, fact_x))
        for i in range(len(nl) - 2, -1, -1):
            for j in range(i + 1, len(nl)):
                if nl[j] >= nl[i]:
                    nl[j] += 1

        return nl

    @staticmethod
    def __get_candidates(num_loops: int) -> list[list[int]]:
        """Get all 1c 2c 3c possible interchanges for `num_loops`

        Args:
            num_loops (int): The number of loops in the operation.

        Returns:
            list[tuple]: The list of all possible interchanges.
        """

        interchanges = []
        for c in [1, 2, 3]:
            level_interchanges = []
            for _ in range(Config().max_num_loops - c):
                level_interchanges.append(list(range(num_loops)))
            for i in range(num_loops - c):
                params = list(range(num_loops))
                params[i], params[i + c] = params[i + c], params[i]
                level_interchanges[i] = params
            interchanges += level_interchanges
        return interchanges

    @classmethod
    def incomplete_interchange(cls, state: OperationState) -> Optional['Interchange']:
        if not state.has_incomplete_action:
            return None
        incomplete_action = state.latest_action

        if not isinstance(incomplete_action, Interchange):
            return None

        return incomplete_action
