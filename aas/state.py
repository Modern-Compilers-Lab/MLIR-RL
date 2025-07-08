from aas import config as cfg
from aas.observation.operation import OperationFeatures, formula_str_to_list
from aas.observation.benchmark import BenchmarkFeatures
from aas.action import Action, Parallelization, Vectorization, NoTransformation, ParameterizedAction, Interchange
import torch
import math


class OperationState:
    """Class to represent the operation state."""
    bench_features: BenchmarkFeatures
    """Benchmark features for the benchmark that the operation is part of."""
    operation_tag: str
    """Tag used to identify the operation in the MLIR code."""
    operation_features: OperationFeatures
    """Features of the operation."""
    step_count: int
    """The current step in the list of transformations applied to the operation."""
    transformation_history: list[Action]
    """List of transformations with their parameters applied to the operation."""

    def __init__(self, bench_features: str, operation_tag: str, operation_features: OperationFeatures,
                 step_count: int, transformation_history: list[Action]):
        """Initialize the operation state."""
        self.bench_features = bench_features
        self.operation_tag = operation_tag
        self.operation_features = operation_features
        self.step_count = step_count
        self.transformation_history = transformation_history

    def get_tensor_dim():
        """Get the tensor dimension of the operation state.

        Returns:
            int: The observation tensor dimension.
        """
        # TODO: Update this when adding new transformations
        L = cfg.max_num_loops
        SL = cfg.max_num_stores_loads
        LSD = cfg.max_num_load_store_dim
        TS = cfg.num_tile_sizes
        NT = cfg.num_transformations
        return 6 + 5 + 1 + 1 + L * 2 + SL * LSD * L + LSD * L + NT + L * (TS + 1)

    def to_tensor(self):
        """Convert the operation state to a torch tensor.

        Returns:
            torch.Tensor: The operation state as a torch tensor.
        """

        # Disable gradient computation
        with torch.no_grad():
            # Operation type (size=6)
            operation_type_ohe = torch.zeros(6)
            if self.operation_features.operation_type == 'matmul':
                operation_type_ohe[0] = 1
            elif 'conv_2d' in self.operation_features.operation_type:
                operation_type_ohe[1] = 1
            elif self.operation_features.operation_type == 'pooling':
                operation_type_ohe[2] = 1
            elif self.operation_features.operation_type == 'add':
                operation_type_ohe[3] = 1
            elif self.operation_features.operation_type == 'generic':
                operation_type_ohe[4] = 1
            else:
                operation_type_ohe[5] = 1

            # Operations count (size=5)
            operations_count = torch.tensor(list(self.operation_features.op_count.values()))

            # Operation iteration space size (size = 1)
            op_iter_space_size = torch.tensor([self.operation_features.op_iter_space_size])
            if cfg.normalize_features:
                op_iter_space_size = torch.log2(op_iter_space_size)

            # Vectorizable (size = 1)
            vectorizable = torch.tensor([1 if self.operation_features.vectorizable else 0])

            # Nested loop features: (upper bound, iter type) (size = max_num_loops * 2)
            indices = [nested_loop.arg for nested_loop in self.operation_features.nested_loops]
            indices_dim = {arg: i for (i, arg) in enumerate(indices)}
            nested_loops = torch.zeros((cfg.max_num_loops, 2))
            for i, nested_loop in enumerate(self.operation_features.nested_loops):
                if i == cfg.max_num_loops:
                    break
                nested_loops[i, 0] = math.log2(nested_loop.upper_bound) if cfg.normalize_features else nested_loop.upper_bound
                nested_loops[i, 1] = 1 if nested_loop.iterator_type == 'parallel' else 0

            # Load access matrices (size = max_num_stores_loads * max_num_load_store_dim * max_num_loops)
            load_data = self.operation_features.load_data
            load_access_matrices = torch.zeros((cfg.max_num_stores_loads, cfg.max_num_load_store_dim, cfg.max_num_loops))
            for load_i, load in enumerate(load_data):
                if load_i == cfg.max_num_stores_loads:
                    break
                dimensions_terms = [formula_str_to_list(term) for term in load]
                for m, dimension_term in enumerate(dimensions_terms):
                    for index, factor in dimension_term:
                        if index in indices_dim:
                            n = indices_dim[index]
                            load_access_matrices[load_i, m, n] = factor

            # Store access matrices (size = max_num_load_store_dim * max_num_loops)
            store_data = self.operation_features.store_data
            store_access_matrices = torch.zeros((cfg.max_num_load_store_dim, cfg.max_num_loops))
            dimensions_terms = [formula_str_to_list(term) for term in store_data]
            for m, dimension_term in enumerate(dimensions_terms):
                for index, factor in dimension_term:
                    n = indices_dim[index]
                    store_access_matrices[m, n] = factor

            # TODO: Update this when adding new transformations
            # Action history (size = num_transformations)
            action_history = torch.zeros(cfg.num_transformations)
            for action in self.transformation_history:
                action_history[action.id] = 1
            # Parallelization history (size = max_num_loops * (num_tile_sizes + 1))
            parallelization_history = torch.zeros((cfg.max_num_loops, cfg.num_tile_sizes + 1))
            for action in self.transformation_history:
                if isinstance(action, Parallelization):
                    for i in range(cfg.max_num_loops):
                        param = action.params[i]
                        if param is None:
                            assert i < len(self.operation_features.nested_loops), "Parallelization parameter for a loop level that excedes the number of loops for an operation cannot be unknown (None)."
                            param_ids = [Parallelization.get_param_id(tile_size) for tile_size in Parallelization.get_tiling_candidates(self.operation_features, loop_id=i)]
                            for j in param_ids:
                                parallelization_history[i, j] = 1 / len(param_ids)
                        else:
                            idx = Parallelization.get_param_id(param)
                            parallelization_history[i, idx] = 1
            # Reshape tensors if needed
            nested_loops = nested_loops.reshape(-1)
            load_access_matrices = load_access_matrices.reshape(-1)
            store_access_matrices = store_access_matrices.reshape(-1)
            parallelization_history = parallelization_history.reshape(-1)

            # Concatenate the feature vectors
            feature_vector = torch.concatenate([
                operation_type_ohe,
                operations_count,
                op_iter_space_size,
                vectorizable,
                nested_loops,
                load_access_matrices,
                store_access_matrices,
                action_history,
                parallelization_history
            ])

        return feature_vector

    def next(self, action: Action):
        """Get the next state of the environment given an action.

        Args:
            action (Action): The action to apply to the current state.

        Returns:
            OperationState: The next state of the environment.
        """
        # if self.is_terminal():
        #     raise ValueError("Cannot apply action to terminal state.")
        last_action = self.transformation_history[-1] if self.transformation_history else None
        if last_action is not None and not action.is_root:
            if isinstance(last_action, ParameterizedAction) and isinstance(action, ParameterizedAction):
                step_count = self.step_count
                new_transformation_history = self.transformation_history.copy()
                new_transformation_history[-1] = action
            else:
                raise ValueError("Cannot apply non-root action while that action or the previous action is not parameterized.")
        else:
            step_count = self.step_count + 1
            new_transformation_history = self.transformation_history + [action]

        op_features = self.operation_features
        if isinstance(action, Interchange):
            # Interchange action updates the operation features
            new_op_features = action.update_op_features(self.operation_features)
            op_features = new_op_features

        return OperationState(
            bench_features=self.bench_features,
            operation_tag=self.operation_tag,
            operation_features=op_features,
            step_count=step_count,
            transformation_history=new_transformation_history
        )

    def is_terminal(self):
        """Check if the state is terminal.

        Returns:
            bool: True if the state is terminal, False otherwise.
        """
        transformation_names = [action.name for action in self.transformation_history]
        if (Vectorization.DEFAULT_NAME in transformation_names) or (NoTransformation.DEFAULT_NAME in transformation_names):
            # If vectorization or no transformation is already applied, the state is terminal
            return True
        elif Parallelization.DEFAULT_NAME in transformation_names:
            # If parallelization is already applied and vectorization is not possible, the state is terminal
            parallel_action = next(action for action in self.transformation_history if isinstance(action, Parallelization))
            new_op_features = parallel_action.update_op_features(self.operation_features)
            return not any([param is None for param in parallel_action.params]) and not Vectorization.is_possible(new_op_features)
        # Otherwise, the state is not terminal
        return False

    def is_final(self):
        """Check if the state is final.

        Returns:
            bool: True if the state is final, False otherwise.
        """
        return self.operation_tag == self.bench_features.operation_tags[0] if cfg.optimization_mode == 'all' else self.operation_tag == self.bench_features.operation_tags[-1]

    def __repr__(self):
        return f"OperationState(bench_name={self.bench_features.bench_name}, operation_tag={self.operation_tag}, " \
               f"operation_features={self.operation_features}, step_count={self.step_count}, " \
               f"transformation_history={self.transformation_history})"
