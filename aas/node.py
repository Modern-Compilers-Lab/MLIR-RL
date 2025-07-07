from aas import config as cfg
from aas.state import OperationState
from aas.action import Action, Parallelization, Vectorization, NoTransformation
from typing import Optional
import math
import random


class Node:
    """Class to represent a node in the MCTS tree"""

    state: OperationState
    """The state of the operation at this node"""
    parent: Optional['Node']
    """The parent node of this node"""
    root_parent: Optional['Node']
    """The node that follows the root action for this node if this node follows a non root action."""
    children: list['Node']
    """The children nodes of this node"""
    nb_visits: int
    """The number of times this node has been visited"""
    q: float
    """The max expected speedup of this node if mode is 'VEMS' or the average expected speedup if mode is 'VEAS'."""
    node_exploration_factor: float
    """The exploration factor for selecting this node from its parent node."""

    def __init__(self, state: OperationState, node_exploration_factor: float = 1.0, parent: Optional['Node'] = None, root_parent: Optional['Node'] = None):
        """Initialize a new node in the MCTS tree.

        Args:
            state (OperationState): The state of the operation at this node.
            node_exploration_factor (float): The exploration factor for selecting this node from its parent node. Defaults to 1.0.
            parent (Optional[Node]): The parent node of this node. Defaults to None.
            root_parent (Optional[Node]): The node that follows the root action for this node if this node follows a non root action. Defaults to None.
        """
        self.state = state
        self.parent = parent
        self.root_parent = root_parent
        self.children = []
        self.nb_visits = 0
        self.q = 0.0
        self.node_exploration_factor = node_exploration_factor

    def add_child(self, child_state: OperationState, child_node_factor: float, child_action: Action):
        """Add a child node to this node.

        Args:
            child_state (OperationState): The state of the operation at the child node.
            child_node_factor (float): The exploration factor for selecting the child node from this node.
            child_action (Action): The action that was applied to the state to create the child state.

        Returns:
            Node: The child node that was added.
        """
        root_parent = None
        if not child_action.is_root:
            curr_last_action = self.state.transformation_history[-1] if self.state.transformation_history else None
            if curr_last_action is not None:
                if curr_last_action.is_root:
                    # If the current last action is a root action, set the root parent to this node
                    root_parent = self
                elif self.root_parent is not None:
                    # If the current last action is not a root action, set the root parent to the root parent of this node
                    root_parent = self.root_parent
                else:
                    # If the current last action is not a root action and this node has no root parent, raise an exception
                    raise ValueError("A non root action node must have a root parent.")
            else:
                raise ValueError("A non root action cannot be the first action to apply. It must follow a root action.")
        # Create child node
        child = Node(
            state=child_state,
            node_exploration_factor=child_node_factor,
            parent=self,
            root_parent=root_parent
        )
        # Add child to children list
        self.children.append(child)
        return child

    def update(self, value: float):
        """Update the node with the given value that was backpropagated during MCTS backpropagation phase.

        Args:
            value (float): The value to update the node with.
        """
        # Update number of visits
        self.nb_visits += 1
        # Get Q value
        self.q = (self.q * (self.nb_visits - 1) + value) / self.nb_visits

    def update_leaf(self, new_value: float):
        """Update the s score of the node with a new value.

        Args:
            new_value (float): The new value to update the s score with.
            mode (Literal['VEMS', 'VEAS']): The update mode. 'VEMS' for max expected speedup and 'VEAS' for average expected speedup.
        """
        # If node is not a leaf, raise an exception
        if self.children:
            raise Exception("The current node is not a leaf as it has children.")
        # Update Q value
        self.q = new_value

    def update_visits_count(self):
        """Increments the number of visits of the node."""
        self.nb_visits += 1

    def get_s_score(self, min_value: float, max_value: float, c_puct: float = 1.0, noise: Optional[float] = None) -> float:
        """Get the s score of the node.

        Args:
            min_value (float): The minimum value of the tree.
            max_value (float): The maximum value of the tree.
            c_puct (float): The exploration parameter for the PUCT formula. Defaults to 1.0.
            noise (Optional[float]): The noise to add to the node exploration factor. Defaults to None.

        Returns:
            float: The s score of the node.
        """
        # Initialize Q value
        q_hat = self.q
        # Normalize Q value to [-1, 1]
        if max_value != min_value:
            q_hat = 2.0 * ((q_hat - min_value) / (max_value - min_value)) - 1.0
        # Calculate S score
        if self.parent is not None:
            # If the node is not the root node, calculate and add the PUCT formula
            p = self.node_exploration_factor
            if noise is not None:
                p = 0.75 * p + 0.25 * noise
            return q_hat + c_puct * p * ((math.sqrt(self.parent.nb_visits) / (1 + self.nb_visits)))
        else:
            # If the node is the root node, return the Q value
            return q_hat

    def get_available_actions(self) -> list[Action]:
        """Get the available actions from this node.

        Returns:
            list[Action]: The list of available actions.
        """
        transformation_names = [action.name for action in self.state.transformation_history]
        if (Vectorization.DEFAULT_NAME in transformation_names) or (NoTransformation.DEFAULT_NAME in transformation_names):
            # If vectorization or no transformation is already applied, return an empty list
            return []
        elif Parallelization.DEFAULT_NAME in transformation_names:
            parallel_action = next(action for action in self.state.transformation_history if isinstance(action, Parallelization))
            if any([param is None for param in parallel_action.params]):
                # If any parallelization parameter is None, continue the hierarchical search for tiling candidates
                loop_id = random.choice([i for i, param in enumerate(parallel_action.params) if param is None])
                candidates = Parallelization.get_tiling_candidates(self.state.operation_features, loop_id=loop_id)
                new_actions = []
                for candidate in candidates:
                    new_params = parallel_action.params.copy()
                    new_params[loop_id] = candidate
                    new_actions.append(Parallelization(new_params, is_root=False))
                return new_actions
            else:
                # If parallelization is already applied, return vectorization if possible or no transformation
                new_op_features = parallel_action.update_op_features(self.state.operation_features)
                return ([Vectorization(), NoTransformation()] if Vectorization.is_possible(new_op_features) else [])
        else:  # NOTE: This should only happen if transformation list is empty
            if cfg.mcts_hierarchical:
                nb_loops = len(self.state.operation_features.nested_loops)
                loop_id = random.randint(0, nb_loops - 1)
                # Get tiling candidates for the loop_id
                candidates = Parallelization.get_tiling_candidates(self.state.operation_features, loop_id=loop_id)
                # Generate tiling parameters
                combinations = []
                for candidate in candidates:
                    combination = [None] * nb_loops + [0] * (cfg.max_num_loops - nb_loops)
                    combination[loop_id] = candidate
                    combinations.append(combination)
            else:
                # Get tiling candidates and number of possible combinations
                nb_combinations = 1
                candidates = Parallelization.get_tiling_candidates(self.state.operation_features)
                for sub_candidates in candidates:
                    nb_combinations *= len(sub_candidates)
                # Generate tiling combinations
                combinations = []
                if cfg.mcts_max_nb_children < 0:
                    # Get all possible tiling combinations
                    combinations = Parallelization.generate_tiling_combinations(candidates)
                    # # Shuffle the combinations for more randomness (encourages exploration)
                    # random.shuffle(combinations)
                else:
                    # Get a random subset of possible tiling combinations
                    seen_combination_ids = set()
                    for _ in min(nb_combinations, cfg.mcts_max_nb_children):
                        combination_id = random.randint(0, nb_combinations - 1)
                        while combination_id in seen_combination_ids:
                            combination_id = random.randint(0, nb_combinations - 1)
                        seen_combination_ids.add(combination_id)
                        combination = []
                        tmp_comb_id = combination_id
                        for sub_candidates in reversed(candidates):
                            combination = sub_candidates[tmp_comb_id % len(sub_candidates)] + combination
                            tmp_comb_id //= len(sub_candidates)
                        combinations.append(combination)

            return [Parallelization(combination) for combination in combinations] + ([Vectorization()] if Vectorization.is_possible(self.state.operation_features) else [])  # + [NoTransformation()]

    def is_terminal(self):
        """Check if the node is a terminal node.

        Returns:
            bool: True if the node is terminal, False otherwise.
        """
        return self.state.is_terminal()

    def to_str(self, min_value: float, max_value: float, c_puct: float = 1.0) -> str:
        """Get a string representation of the node.

        Args:
            c_puct (float): The exploration parameter for the PUCT formula. Defaults to 1.0.
            random_exploration_temperature (float): The temperature for random exploration. Defaults to 0.0.

        Returns:
            str: The string representation of the node.
        """
        return f"<Node nb_visits={self.nb_visits} s={self.get_s_score(min_value, max_value, c_puct=c_puct)} q={self.q} node_p={self.node_exploration_factor} action={self.state.transformation_history[-1] if self.state.transformation_history else None}>"
