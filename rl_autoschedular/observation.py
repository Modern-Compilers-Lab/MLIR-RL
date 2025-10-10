from rl_autoschedular.actions import ActionSpace
from rl_autoschedular.state import OperationState, OperationType, IteratorType, OperationFeatures
import torch
import math

from utils.config import Config

L = Config().max_num_loops
LSD = Config().max_num_load_store_dim
LS = Config().max_num_stores_loads


class ObservationPart:
    @classmethod
    def size(cls) -> int:
        raise NotImplementedError

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        """Create the observation part from the current state."""
        raise NotImplementedError


class OpFeatures(ObservationPart):
    """Class representing operation features in the observation"""

    arith_ops = ['+', '-', '*', '/', 'exp']

    @classmethod
    def size(cls) -> int:
        return len(OperationType) + L + L + LS * LSD * L + LS * LSD * L + len(cls.arith_ops)

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return cls._from_features(state.original_operation_features)

    @classmethod
    def _from_features(cls, op_features: OperationFeatures) -> torch.Tensor:
        indices_dim = {nested_loop.arg: i for i, nested_loop in enumerate(op_features.nested_loops)}

        # Operation type
        op_type = torch.tensor([op_features.operation_type == ot for ot in OperationType])

        # Nested loop features: (upper bounds, iterator types)
        nested_loops = torch.zeros(L)
        iterator_types = torch.zeros(L)
        for i, nested_loop in enumerate(op_features.nested_loops):
            if i == L:
                break
            ub = nested_loop.upper_bound
            match Config().normalize_bounds:
                case 'max':
                    ub = ub / 4096
                case 'log':
                    ub = math.log2(ub)
            nested_loops[i] = ub
            iterator_types[i] = nested_loop.iterator_type == IteratorType.Parallel

        # # Vectorizable
        # vectorizable = torch.tensor([op_features.vectorizable])

        # load access matrices:
        load_access_matrices = torch.zeros((LS, LSD, L))

        for load_i, load in enumerate(op_features.load_data):
            if load_i == LS:
                break
            dimensions_terms = [cls.__formula_str_to_list(term) for term in load]
            for m, dimension_term in enumerate(dimensions_terms):
                if m == LSD:
                    break
                for index, factor in dimension_term:
                    if index not in indices_dim:
                        continue
                    n = indices_dim[index]
                    if n >= L:
                        continue
                    load_access_matrices[load_i, m, n] = factor

        # store access matrices:
        store_access_matrices = torch.zeros((LS, LSD, L))

        for store_i, store in enumerate(op_features.store_data):
            if store_i == LS:
                break
            dimensions_terms = [cls.__formula_str_to_list(term) for term in store]
            for m, dimension_term in enumerate(dimensions_terms):
                if m == LSD:
                    break
                for index, factor in dimension_term:
                    if index not in indices_dim:
                        continue
                    n = indices_dim[index]
                    if n >= L:
                        continue
                    store_access_matrices[store_i, m, n] = factor

        # Operations count:
        operations_count = torch.tensor([op_features.op_count[s] for s in cls.arith_ops])

        feature_vector = torch.cat((
            op_type,
            nested_loops,
            iterator_types,
            # vectorizable,
            load_access_matrices.reshape(-1),
            store_access_matrices.reshape(-1),
            operations_count
        ))

        return feature_vector

    @staticmethod
    def __formula_str_to_list(formula: str) -> list[tuple[str, int]]:
        """Turns assignement formula to a list of (index, factor)
        Example:
            formula = "%x1 - %x2 + %x3 * 5 - %x5 * 3"
            return [('%x1', 1), ('%x2', -1), ('%x3', 5), ('%x5', -3)]

        Args:
            formula (str): the formula as a string input

        Returns:
            list: list of (index, factor) pairs
        """
        formula = formula + ' +'
        terms = formula.split(' ')

        running_factor = 1
        running_term = None

        save = []

        for term in terms:

            if term.startswith('%'):
                running_term = term
            elif term == '+':
                save.append((running_term, running_factor))
                running_factor = 1
            elif term == '-':
                save.append((running_term, running_factor))
                running_factor = -1
            elif term.isnumeric():
                running_factor *= int(term)

        if save[0][0] is None:
            save = save[1:]

        return save


class ProducerOpFeatures(OpFeatures):
    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        if state.producer_features:
            return cls._from_features(state.producer_features)

        return torch.zeros(cls.size())


class ActionHistory(ObservationPart):
    """Class representing action history in the observation"""

    @classmethod
    def size(cls) -> int:
        return ActionSpace.cumulative_history_sizes()[-1]

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return ActionSpace.action_history(state.current_history)


class ActionMask(ObservationPart):
    """Class representing action mask in the observation"""

    @classmethod
    def size(cls) -> int:
        return ActionSpace.cumulative_mask_sizes()[-1]

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return ActionSpace.action_mask(state)


class NumLoops(ObservationPart):
    """Class representing number of loops in the observation"""

    @classmethod
    def size(cls) -> int:
        return 1

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        return torch.tensor([len(state.operation_features.nested_loops)])


class Observation:
    """Class to manage creation and use of observations"""

    parts: list[type[ObservationPart]] = [
        OpFeatures,
        ProducerOpFeatures,
        ActionHistory,
        NumLoops,
        ActionMask
    ]

    @classmethod
    def cumulative_sizes(cls) -> list[int]:
        """Get cumulative sizes of all observation parts."""
        sizes = [0]
        for part in cls.parts:
            sizes.append(sizes[-1] + part.size())
        return sizes

    @classmethod
    def part_number(cls, part: type[ObservationPart]) -> int:
        """Get the index of a part in the observation."""
        return cls.parts.index(part)

    @classmethod
    def get_part(cls, obs: torch.Tensor, part: type[ObservationPart], squeeze: bool = True) -> torch.Tensor:
        """Get a specific part of the observation."""
        part_idx = cls.part_number(part)
        cum_sizes = cls.cumulative_sizes()
        start = cum_sizes[part_idx]
        if part.size() == 1 and squeeze:
            return obs[:, start]
        end = cum_sizes[part_idx + 1]
        return obs[:, start:end]

    @classmethod
    def get_parts(cls, obs: torch.Tensor, *parts: type[ObservationPart]) -> torch.Tensor:
        """Get multiple parts of the observation in a single tensor."""
        return torch.cat([cls.get_part(obs, part, False) for part in parts], dim=1)

    @classmethod
    def from_state(cls, state: OperationState) -> torch.Tensor:
        """Create the full observation from the current state."""
        obs_parts = [part.from_state(state) for part in cls.parts]
        return torch.cat(obs_parts).unsqueeze(0)

    @classmethod
    def from_states(cls, states: list[OperationState]) -> torch.Tensor:
        """Create the full observation for all the states."""
        return torch.cat([cls.from_state(s) for s in states])
