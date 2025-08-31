from utils.log import print_alert
from .tiling import Tiling
from .tiled_parallelization import TiledParallelization
from rl_autoschedular.transforms import transform_TF
from rl_autoschedular.state import BenchmarkFeatures, OperationFeatures, OperationState
from typing import Optional


class TiledFusion(TiledParallelization):
    """Class representing Tiled Fusion action"""

    symbol = 'TPF'

    # --- extras ---
    producer_tag: str

    def __init__(
        self,
        parameters: list[int],
        state: Optional[OperationState] = None,
        producer_tag: Optional[str] = None,
        **extras
    ):
        if (state is None) == (producer_tag is None):
            raise ValueError("Either state or producer tag must be provided and not both")
        if state:
            producer_tag = state.producer_tag
        super().__init__(parameters, state, producer_tag=producer_tag, **extras)

        self.producer_tag = producer_tag

    def __str__(self):
        return f"{self.symbol}({self.producer_tag};{','.join(map(str, self.parameters))})"

    @property
    def consumer_tag(self):
        return self.operation_tag

    @property
    def new_producer_tag(self):
        return f'{self.producer_tag}_{self.consumer_tag}'

    def is_tag_fused(self, prod_tag: str):
        return prod_tag.endswith('_' + self.consumer_tag)

    @classmethod
    def is_allowed(cls, state):
        already_tiled = any(
            isinstance(action, Tiling) for action in
            state.operation_features.pre_actions + state.transformation_history[0]
        )
        has_producers = state.producer_tag is not None

        return has_producers and not already_tiled

    def _apply_ready(self, code):
        return transform_TF(
            code,
            self.consumer_tag,
            self.producer_tag,
            self.new_producer_tag,
            self.tiling_params,
            self.parallel_params,
        )

    def update_features(self, operation_features: OperationFeatures):
        new_operation_features = operation_features.copy()
        for i, (prod, prod_idx) in enumerate(operation_features.producers):
            if prod != self.producer_tag:
                continue
            new_operation_features.producers[i] = (self.new_producer_tag, prod_idx)

        return super().update_features(new_operation_features)

    def update_producer_features(self, state: OperationState, bench_feats: BenchmarkFeatures):
        """Update the features of the prducer after the fusion.

        Note:
            - This update modifies the bench features inplace
            - Currently we only support having one use in the containing op
        """
        prod_feats = state.producer_features.copy()

        self.__update_consumers_list(prod_feats, state)

        self.__record_implicit_tiling(prod_feats, state)

        # Insert the new producer in benchmark features
        insert_idx = self.__get_insertion_position(state, bench_feats)
        bench_feats.operations[self.new_producer_tag] = prod_feats
        bench_feats.operation_tags.insert(insert_idx, self.new_producer_tag)

        self.__handle_producer_original_op(bench_feats)

    def __get_insertion_position(self, state: OperationState, bench_feats: BenchmarkFeatures):
        prod_oprnd_idx = state.producer_operand_idx
        assert prod_oprnd_idx is not None

        for other_prod, other_prod_idx in sorted(state.operation_features.producers, key=lambda p: p[1]):
            if not self.is_tag_fused(other_prod):
                continue
            if prod_oprnd_idx == other_prod_idx:
                assert other_prod == self.new_producer_tag
                continue
            if prod_oprnd_idx < other_prod_idx:
                return bench_feats.operation_tags.index(other_prod)

        return bench_feats.operation_tags.index(self.consumer_tag)

    def __handle_producer_original_op(self, bench_feats: BenchmarkFeatures):
        prod_original_feats = bench_feats.operations[self.producer_tag]
        if len(set(c for c, _ in prod_original_feats.consumers)) <= 1:
            # If producer doesn't have other consumers -> remove original op
            assert prod_original_feats.consumers and prod_original_feats.consumers[0][0] == self.consumer_tag, \
                'Consumer not found in producer features'
            del bench_feats.operations[self.producer_tag]
            bench_feats.operation_tags.remove(self.producer_tag)
        else:
            # Else -> remove this consumer from the original op
            prod_original_feats.consumers = [
                (c, i) for c, i in prod_original_feats.consumers
                if c != self.consumer_tag
            ]

    def __update_consumers_list(self, prod_feats: OperationFeatures, state: OperationState):
        # Fused producer has only this op as a consumer
        consumer_in_prod = [(c, i) for c, i in prod_feats.consumers if c == self.consumer_tag]
        prod_in_consumer = [(p, i) for p, i in state.operation_features.producers if p == self.new_producer_tag]
        relations = [(res_i, operand_i) for (_, res_i), (_, operand_i) in zip(consumer_in_prod, prod_in_consumer)]
        if len(relations) == 1:
            assert state.producer_operand_idx == relations[0][1]
            chosen_relation = relations[0]
        else:
            print_alert(
                "Having multiple uses isn't currently supported\n"
                "-> Considering only first use"
            )
            chosen_relation = [(res_i, operand_i) for (res_i, operand_i) in relations if operand_i == state.producer_operand_idx][0]
        prod_feats.consumers = [(self.consumer_tag, chosen_relation[0])]

    def __record_implicit_tiling(self, prod_feats: OperationFeatures, state: OperationState):
        prod_oprnd_idx = state.producer_operand_idx
        consumer_feats = state.operation_features
        assert prod_oprnd_idx is not None

        # 1. Get producer result tiling sizes
        prod_load = (consumer_feats.load_data + consumer_feats.store_data)[prod_oprnd_idx]
        consumer_args_tile_sizes = {nl.arg: self.parameters[i] for i, nl in enumerate(consumer_feats.nested_loops)}
        prod_res_tile_sizes: list[int] = []
        for dim_pos, dim_str in enumerate(prod_load):
            dim_str = dim_str.strip()
            dim_new_terms = []
            for term in dim_str.split(' '):
                if term not in consumer_args_tile_sizes:
                    dim_new_terms.append(term)
                    continue
                # We need the last index not the size
                dim_new_terms.append(consumer_args_tile_sizes[term] - 1)
            last_idx_str = ' '.join(dim_new_terms)

            try:
                last_idx = int(eval(last_idx_str))
                prod_res_tile_sizes.append(last_idx + 1)
            except Exception:
                raise Exception(f"Unsupported producer load [{prod_load}] at position {dim_pos}")

        # 2. Get actual tile sizes of producer
        prod_res_store = prod_feats.store_data[prod_feats.consumers[0][1]]
        assert len(prod_res_store) == len(prod_res_tile_sizes), "Unexpected dimensions for producer result tile sizes" \
            f", expected {len(prod_res_store)} but found {len(prod_res_tile_sizes)}"

        tile_sizes = [0 for _ in prod_feats.nested_loops]
        prod_args_dims = {nl.arg: i for i, nl in enumerate(prod_feats.nested_loops)}
        for dim_pos, dim_str in enumerate(prod_res_store):
            dim_str = dim_str.strip()
            if dim_str not in prod_args_dims:
                raise Exception(f"Unsupported producer store [{prod_res_store}] at position {dim_pos}")
            tile_sizes[prod_args_dims[dim_str]] = prod_res_tile_sizes[dim_pos]

        # 3. Add the tiling as a pre-action in the producer
        pre_tiling = TiledParallelization(
            tile_sizes,
            operation_tag=self.new_producer_tag,
            iterators=[nl.iterator_type.value for nl in prod_feats.nested_loops],
        )
        prod_feats.pre_actions.append(pre_tiling)
