"""Base action classes for MLIR loop transformations.

This module defines the abstract base class for transformation actions and provides
the action interface that all concrete transformation actions must implement.
"""

from typing import Optional, overload, Union, Any
from rl_autoschedular_paper_transformer.state import OperationState, OperationFeatures
import torch
from torch.distributions import Distribution
from mlir._mlir_libs._mlir.ir import Module  # type: ignore


class Action:
    """Base action class"""

    symbol: str

    operation_tag: str
    parameters: Optional[list[int]]
    extras: dict[str, Any]

    # --- defaults ---
    ready: bool = True
    terminal: bool = False
    sub_actions: list['Action'] = []

    @overload
    def __init__(self, *, operation_tag: str, **extras):
        """Initialize action without parameters"""
        ...

    @overload
    def __init__(self, state: OperationState, /, **extras):
        """Initialize action dependent on state but without parameters

        Args:
            state: current state to apply the action on
        """
        ...

    @overload
    def __init__(self, parameters: list[int], /, *, operation_tag: str, **extras):
        """Initialize action with parameters

        Args:
            parameters: list of parameters for the action
        """
        ...

    @overload
    def __init__(self, parameters: list[int], state: OperationState, /, **extras):
        """Initialize action with unprocessed parameters

        Args:
            parameters: list of parameters for the action
            state: current state to apply the action on
        """
        ...

    def __init__(
        self,
        arg1: Optional[Union[OperationState, list[int]]] = None,
        arg2: Optional[OperationState] = None,
        /, *,
        operation_tag: Optional[str] = None,
        **extras
    ):
        if isinstance(arg1, OperationState):
            parameters = None
            state = arg1
        else:
            parameters = arg1
            state = arg2
        if (state is None) == (operation_tag is None):
            raise ValueError("Either state or operation tag must be provided and not both")
        if state:
            operation_tag = state.operation_tag
        self.operation_tag = operation_tag
        self.parameters = parameters
        self.extras = {'operation_tag': operation_tag, **extras}
        if 'process_params' in self.extras:
            del self.extras['process_params']

    def __repr__(self) -> str:
        """String representation of the action with extra params"""
        params_list = list(map(str, self.parameters)) if self.parameters else []
        params_list.extend(f'{k} = {v}' for k, v in self.extras.items())

        return f"{self.__class__.__name__}({', '.join(params_list)})"

    def __str__(self) -> str:
        """String representation of the action"""
        return f"{self.symbol}({','.join(map(str, self.parameters)) if self.parameters else ''})"

    @classmethod
    def from_str(cls, state: OperationState, action_str: str) -> 'Action':
        """Create an action from a string representation

        Args:
            state: current state to apply the action on
            action_str: string representation of the action

        Returns:
            action created from the string representation
        """
        symbol = action_str.split('(')[0]
        if symbol != cls.symbol:
            raise ValueError(f'Symbol mismatch for class {cls.__name__}: {symbol} != {cls.symbol}')

        parameters = list(map(int, action_str.split('(')[1].split(')')[0].split(',')))
        if not parameters:
            return cls(state)
        return cls(parameters, state, process_params=False)

    @classmethod
    def params_size(cls) -> int:
        """Return the size of the parameters in the index for this action type

        Returns:
            size of the parameters for this action type
        """
        return 0

    @classmethod
    def network_output_size(cls) -> int:
        """Return the size of the network output for this action type

        Returns:
            size of the network output for this action type
        """
        return 0

    @classmethod
    def mask_size(cls) -> int:
        """Return the size of the mask for this action type

        Returns:
            size of the mask for this action type
        """
        return cls.network_output_size()

    @classmethod
    def history_size(cls) -> int:
        """Return the size of the history for this action type

        Returns:
            size of the history for this action type
        """
        return 0

    @classmethod
    def is_allowed(cls, state: OperationState) -> bool:
        """Check if this action type is allowed in the current state

        Args:
            state: current state to check the action on

        Returns:
            True if the action is allowed, False otherwise
        """
        return True

    @classmethod
    def action_mask(cls, state: OperationState) -> Optional[torch.Tensor]:
        """Return the action mask for this action type in the current state

        Args:
            state: current state to check the action on

        Returns:
            action mask for this action type, or None if not applicable
        """
        return None

    @classmethod
    def action_history(cls, seq: list['Action']) -> Optional[torch.Tensor]:
        """Return the action history for this action type in the current state

        Args:
            seq: sequence of actions in the current state

        Returns:
            action history for this action type, or None if not applicable
        """
        return None

    @classmethod
    def distribution(cls, logits: torch.Tensor) -> Distribution:
        """Create a distribution for this action type based on the logits

        Args:
            logits: Logits for the action selection.

        Returns:
            A distribution object for this action type.
        """
        raise NotImplementedError

    @classmethod
    def uniform_distribution(cls, logits: torch.Tensor, num_loops: torch.Tensor) -> Distribution:
        """Create a uniform distribution for this action type based on the logits and number of loops

        Args:
            logits: Logits for the action selection.
            num_loops: Number of loops in the operation state.

        Returns:
            A uniform distribution object for this action type.
        """
        return cls.distribution(logits)

    @classmethod
    def distribution_stats(cls, distribution: Distribution, index: torch.Tensor, eps_distribution: Optional[Distribution], eps: Optional[float] = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the log probabilities and entropies for the distribution

        Args:
            distribution: The distribution to calculate stats for.
            eps_distribution: The epsilon distribution for exploration.
            index: The params index.
            eps: Epsilon value for exploration. Defaults to None.

        Returns:
            Log probabilities and entropies.
        """
        raise NotImplementedError

    @classmethod
    def sample(cls, distribution: Distribution, eps_distribution: Distribution, num_loops: torch.Tensor, uniform: bool, greedy: bool) -> torch.Tensor:
        """Sample an action based on the distribution

        Args:
            distribution: The distribution to sample from.
            eps_distribution: The epsilon distribution for exploration.
            num_loops: Number of loops in the operation state.
            uniform: Whether to sample uniformly.
            greedy: Whether to sample greedily.

        Returns:
            Sampled action index.
        """
        raise NotImplementedError

    def apply(self, module: Module) -> Module:
        """Apply action on the current code

        Args:
            module: current code to apply the action on

        Returns:
            the new transformed code
        """
        if not self.ready:
            return

        self._apply_ready(module)

    def _apply_ready(self, module: Module):
        """Apply action that is guarenteed to be ready on the current state

        Args:
            module: current code to apply the action on
        """
        raise NotImplementedError

    def update_features(self, operation_features: OperationFeatures) -> OperationFeatures:
        """Update the operation features based on the action

        Args:
            operation_features: The operation features to update.

        Returns:
            The updated operation features.
        """
        return operation_features
