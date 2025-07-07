from aas import config as cfg
from aas.observation.operation import OperationFeatures
from typing import Optional
import math


class Action:
    """Class to represent a transformation as an agent action."""
    _id: int
    """The ID of the transformation"""
    _name: str
    """The name of the transformation"""
    is_root: bool
    """Whether the action is a root action or not in a hiearchical action setting.
    The root action is the first action with only one parameter set and other parameters are unknown.
    It should be followed by the same action with one more different parameter set and so on until all parameters are set.
    Default is True."""

    def __init__(self, id: int, name: str):
        """Initialize a new action.

        Args:
            id (int): The ID of the transformation.
            name (str): The name of the transformation.
        """
        self._id = id
        self._name = name
        self.is_root = True

    @property
    def id(self):
        """The ID of the transformation."""
        return self._id

    @property
    def name(self):
        """The name of the transformation."""
        return self._name

    def update_op_features(self, operation_features: OperationFeatures):
        """Update the operation features with the transformation.

        Args:
            operation_features (OperationFeatures): The operation features to update.
        """
        ...

    def __repr__(self):
        return f'{self.name}()'


class ParameterizedAction(Action):
    """Class to represent a parameterized transformation as an agent action."""
    params: list[Optional[int]]
    """The parameters of the transformation"""

    def __init__(self, id: int, name: str, params: list[Optional[int]], is_root: bool = True):
        """Initialize a new parameterized action.

        Args:
            id (int): The ID of the transformation.
            name (str): The name of the transformation.
            params (list[Optional[int]]): The parameters of the transformation.
            is_root (bool): Whether the action is a root action or not. Defaults to True.
        """
        super().__init__(id, name)
        self.params = params
        self.is_root = is_root

    def __repr__(self):
        return f'{self.name}({', '.join([str(e) if e is not None else '-' for e in self.params])})'


class Parallelization(ParameterizedAction):
    """Class to represent a parallelization transformation as an agent action."""

    DEFAULT_NAME = 'TP'
    """The default name of the parallelization transformation"""
    ID = 0
    """The ID of the parallelization transformation"""

    def __init__(self, params: list[Optional[int]], is_root: bool = True):
        """Initialize a new parallelization action.

        Args:
            params (list[Optional[int]]): The parameters of the transformation.
            is_root (bool): Whether the action is a root action or not. Defaults to True.
        """
        super().__init__(Parallelization.ID, Parallelization.DEFAULT_NAME, params, is_root=is_root)

    def get_param_id(tile_size: int):
        """Get the ID of the tile size.

        Args:
            tile_size (int): The tile size to get the ID for.

        Returns:
            int: The ID of the parameter.
        """
        return int(math.log2(tile_size)) + 1 if tile_size > 0 else 0

    def get_tile_size(param_id: int):
        """Get the tile size from the parameter ID.

        Args:
            param_id (int): The parameter ID to get the tile size from.

        Returns:
            int: The tile size.
        """
        return 2 ** (param_id - 1) if param_id > 0 else 0

    def generate_tiling_combinations(candidates: list[list[int]]):
        """Generate all possible tiling combinations from the list of candidates.

        Args:
            candidates (list[list[int]]): The list of candidates to generate tiling combinations from.

        Returns:
            list[list[int]]: The list of all possible tiling combinations.
        """
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [[candidate] for candidate in candidates[0]]
        else:
            sub_candidates = Parallelization.generate_tiling_combinations(candidates[1:])
            return [[candidate] + sub_candidate for candidate in candidates[0] for sub_candidate in sub_candidates]

    def get_tiling_candidates(operation_features: OperationFeatures, loop_id: Optional[int] = None):
        """Get the tiling candidates for the operation features.

        Args:
            operation_features (OperationFeatures): The operation features to get the tiling candidates from.
            loop_id (Optional[int], optional): The loop ID to get the tiling candidates for. Defaults to None.

        Returns:
            list[list[int]]: The list of tiling candidates.
        """
        start_id = loop_id if loop_id is not None else 0
        end_id = loop_id + 1 if loop_id is not None else len(operation_features.nested_loops)
        candidates = []
        for nested_loop in operation_features.nested_loops[start_id:end_id]:
            # If upperbound equal 1, we only have candidates of 1
            if nested_loop.upper_bound == 1:
                sub_candidates = [0, 1]
            # If the data format is json and the iterator type is reduction, we don't do tiling
            # TODO: the condition has to change because it's not related to the data format, it's related to a non thread safe tiling problem
            # so we skip it to not let it happen for now
            if cfg.data_format == 'json' and nested_loop.iterator_type == 'reduction':
                sub_candidates = [0]
            else:
                # We take the divisors of the upperbound
                sub_candidates = [0]
                i = 1
                while i <= nested_loop.upper_bound and len(sub_candidates) < cfg.num_tile_sizes + 1:
                    if nested_loop.upper_bound % i == 0:
                        sub_candidates.append(i)
                    i *= 2
            candidates.append(sub_candidates)

        # Fill other loops with 0 as candidate
        if loop_id is None:
            for _ in range(len(operation_features.nested_loops), cfg.max_num_loops):
                candidates.append([0])

        return candidates if loop_id is None else candidates[0]

    def update_op_features(self, operation_features: OperationFeatures):
        """Update the operation features with the tiling.

        Args:
            operation_features (OperationFeatures): The operation features to update.
        """
        op_iter_space_size = 1
        for i, nested_loop in enumerate(operation_features.nested_loops):
            op_iter_space_size *= self.params[i] if self.params[i] else nested_loop.upper_bound
        return OperationFeatures(
            operation_type=operation_features.operation_type,
            op_count=operation_features.op_count,
            op_iter_space_size=op_iter_space_size,
            nested_loops=operation_features.nested_loops,
            load_data=operation_features.load_data,
            store_data=operation_features.store_data,
            vectorizable=operation_features.vectorizable
        )


class Vectorization(Action):
    """Class to represent a vectorization transformation as an agent action."""

    DEFAULT_NAME = 'V'
    """The default name of the vectorization transformation"""
    ID = 1
    """The ID of the vectorization transformation"""

    def __init__(self):
        """Initialize a new vectorization action."""
        super().__init__(Vectorization.ID, Vectorization.DEFAULT_NAME)

    def is_possible(operation_features: OperationFeatures):
        return operation_features.vectorizable and (operation_features.op_iter_space_size <= cfg.vect_size_limit)


class NoTransformation(Action):
    """Class to represent a no transformation action as an agent action."""

    DEFAULT_NAME = 'NT'
    """The default name of the no transformation action"""
    ID = 2
    """The ID of the no transformation action"""

    def __init__(self):
        """Initialize a new no transformation action."""
        super().__init__(NoTransformation.ID, NoTransformation.DEFAULT_NAME)


class Img2Col(Action):
    """Class to represent an image to column transformation as an agent action."""

    DEFAULT_NAME = 'I2C'
    """The default name of the image to column transformation"""
    ID = 3
    """The ID of the image to column transformation"""

    def __init__(self):
        """Initialize a new image to column action."""
        super().__init__(Img2Col.ID, Img2Col.DEFAULT_NAME)

    def is_possible(operation_features: OperationFeatures):
        return operation_features.operation_type == 'conv_2d'


class Tiling(ParameterizedAction):
    """Class to represent a tiling alone transformation as an agent action."""

    DEFAULT_NAME = 'T'
    """The default name of the tiling transformation"""
    ID = 4
    """The ID of the tiling transformation"""

    def __init__(self, params: list[Optional[int]], is_root: bool = True):
        """Initialize a new tiling action.

        Args:
            params (list[Optional[int]]): The parameters of the transformation.
            is_root (bool): Whether the action is a root action or not. Defaults to True.
        """
        super().__init__(Tiling.ID, Tiling.DEFAULT_NAME, params, is_root=is_root)

    def get_param_id(tile_size: int):
        """Get the ID of the tile size.

        Args:
            tile_size (int): The tile size to get the ID for.

        Returns:
            int: The ID of the parameter.
        """
        return int(math.log2(tile_size)) + 1 if tile_size > 0 else 0

    def get_tile_size(param_id: int):
        """Get the tile size from the parameter ID.

        Args:
            param_id (int): The parameter ID to get the tile size from.

        Returns:
            int: The tile size.
        """
        return 2 ** (param_id - 1) if param_id > 0 else 0

    def generate_tiling_combinations(candidates: list[list[int]]):
        """Generate all possible tiling combinations from the list of candidates.

        Args:
            candidates (list[list[int]]): The list of candidates to generate tiling combinations from.

        Returns:
            list[list[int]]: The list of all possible tiling combinations.
        """
        if len(candidates) == 0:
            return []
        elif len(candidates) == 1:
            return [[candidate] for candidate in candidates[0]]
        else:
            sub_candidates = Tiling.generate_tiling_combinations(candidates[1:])
            return [[candidate] + sub_candidate for candidate in candidates[0] for sub_candidate in sub_candidates]

    def get_tiling_candidates(operation_features: OperationFeatures):
        """Get the tiling candidates for the operation features.

        Args:
            operation_features (OperationFeatures): The operation features to get the tiling candidates from.

        Returns:
            list[list[int]]: The list of tiling candidates.
        """
        candidates = []
        for nested_loop in operation_features.nested_loops:
            # If upperbound equal 1, we only have candidates of 1
            if nested_loop.upper_bound == 1:
                sub_candidates = [0, 1]
            # If the data format is json and the iterator type is reduction, we don't do tiling
            # TODO: the condition has to change because it's not related to the data format, it's related to a non thread safe tiling problem
            # so we skip it to not let it happen for now
            if cfg.data_format == 'json' and nested_loop.iterator_type == 'reduction':
                sub_candidates = [0]
            else:
                # We take the divisors of the upperbound
                sub_candidates = [0]
                i = 1
                while i <= nested_loop.upper_bound and len(sub_candidates) < cfg.num_tile_sizes + 1:
                    if nested_loop.upper_bound % i == 0:
                        sub_candidates.append(i)
                    i *= 2
            candidates.append(sub_candidates)
        # Fill other loops with 0 as candidate
        for _ in range(len(operation_features.nested_loops), cfg.max_num_loops):
            candidates.append([0])

        return candidates

    def update_op_features(self, operation_features: OperationFeatures):
        """Update the operation features with the tiling.

        Args:
            operation_features (OperationFeatures): The operation features to update.
        """
        op_iter_space_size = 1
        for i, nested_loop in enumerate(operation_features.nested_loops):
            op_iter_space_size *= self.params[i] if self.params[i] else nested_loop.upper_bound
        return OperationFeatures(
            operation_type=operation_features.operation_type,
            op_count=operation_features.op_count,
            op_iter_space_size=op_iter_space_size,
            nested_loops=operation_features.nested_loops,
            load_data=operation_features.load_data,
            store_data=operation_features.store_data,
            vectorizable=operation_features.vectorizable
        )
