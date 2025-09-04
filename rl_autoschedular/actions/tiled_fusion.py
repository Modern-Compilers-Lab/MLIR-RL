from utils.log import print_alert
from .tiled_parallelization import TiledParallelization
from rl_autoschedular.transforms import transform_TF, transform_tile
from rl_autoschedular.state import BenchmarkFeatures, OperationFeatures, OperationState
from typing import Optional, Union, overload


class TiledFusion(TiledParallelization):
    """Class representing Tiled Fusion action"""

    symbol = 'TPF'

    # --- extras ---
    producer_tag: str
    producer_operand_idx: int
    producer_feats_updated: bool

    @overload
    def __new__(cls, parameters: list[int], state: OperationState, **extras) -> Union[TiledParallelization, 'TiledFusion']:
        ...

    @overload
    def __new__(
        cls,
        parameters: list[int],
        *_,
        producer_tag: str,
        producer_operand_idx: int,
        **extras
    ) -> Union[TiledParallelization, 'TiledFusion']:
        ...

    @overload
    def __new__(cls) -> 'TiledFusion':
        """This is only for pickle, not for any other use"""
        ...

    def __new__(cls, *args, **kwargs):
        if not args and not kwargs:
            return super().__new__(cls)
        trial_instance = TiledParallelization(*args, **kwargs)
        if all(p == 0 for p in trial_instance.parallel_params):
            # This becomes just a tiled parallelization and not a tiled fusion
            return trial_instance

        return super().__new__(cls)

    def __init__(
        self,
        parameters: list[int],
        state: Optional[OperationState] = None,
        producer_tag: Optional[str] = None,
        producer_operand_idx: Optional[int] = None,
        **extras
    ):
        args_is_none = [
            producer_tag is None,
            producer_operand_idx is None
        ]
        if (state is None) in args_is_none:
            raise ValueError("Either state or preprocessing attributes must be provided and not both")
        if state:
            producer_tag = state.producer_tag
            producer_operand_idx = state.producer_operand_idx
        assert producer_tag is not None and producer_operand_idx is not None
        super().__init__(
            parameters,
            state,
            producer_tag=producer_tag,
            producer_operand_idx=producer_operand_idx,
            **extras
        )

        self.producer_tag = producer_tag
        self.producer_operand_idx = producer_operand_idx
        self.producer_feats_updated = False

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
    def is_allowed(cls, state: OperationState):
        has_producers = state.producer_tag is not None

        return has_producers and super().is_allowed(state)

    def _apply_ready(self, code):
        f_code = transform_TF(
            code,
            self.consumer_tag,
            self.producer_tag,
            self.new_producer_tag,
            self.parallel_params,
        )
        return transform_tile(f_code, self.operation_tag, self.tiling_params)

    def update_features(self, operation_features: OperationFeatures):
        if not self.producer_feats_updated:
            raise Exception("Producer features must be updated first")
        new_operation_features = operation_features.copy()
        new_operation_features = super().update_features(new_operation_features)

        # NOTE: To change with mutliple producers support
        # NOTE: To change with mutliple uses support
        new_operation_features.producers = [(self.new_producer_tag, self.producer_operand_idx)]

        return new_operation_features

    def update_producer_features(self, state: OperationState, bench_feats: BenchmarkFeatures):
        """Update the features of the prducer after the fusion.

        Note:
            - This update modifies the bench features inplace
            - Currently we only support having one use in the containing op
        """
        prod_feats = state.producer_features.copy()

        self.__update_consumers_and_producers(prod_feats, state)

        self.__record_implicit_tiling(prod_feats, state)

        self.__insert_in_bench_feats(prod_feats, state, bench_feats)

        self.__handle_producer_original_op(bench_feats)

        self.producer_feats_updated = True

    def __insert_in_bench_feats(self, prod_feats: OperationFeatures, state: OperationState, bench_feats: BenchmarkFeatures):
        # Get insertion position
        insert_idx = None
        for other_prod, other_prod_idx in sorted(state.operation_features.producers, key=lambda p: p[1]):
            if not self.is_tag_fused(other_prod):
                continue
            if self.producer_operand_idx == other_prod_idx:
                assert other_prod == self.producer_tag
                continue
            if self.producer_operand_idx < other_prod_idx:
                insert_idx = bench_feats.operation_tags.index(other_prod)
                break

        if insert_idx is None:
            insert_idx = bench_feats.operation_tags.index(self.consumer_tag)

        # Record new producer and insert its tag
        bench_feats.operations[self.new_producer_tag] = prod_feats
        bench_feats.operation_tags.insert(insert_idx, self.new_producer_tag)

    def __handle_producer_original_op(self, bench_feats: BenchmarkFeatures):
        prod_original_feats = bench_feats.operations[self.producer_tag]
        prod_producers = set(p for p, _ in prod_original_feats.producers)

        # Update producers of original to incluse new producer
        # as a consumer too (for dependence analysis reasons)
        for prod_producer in prod_producers:
            prod_prod_feats = bench_feats.operations[prod_producer]
            prod_prod_feats.consumers.extend([
                (self.new_producer_tag, i) for c, i in
                prod_prod_feats.consumers if c == self.producer_tag
            ])

        # Remove or update dependents of original op
        if len(set(c for c, _ in prod_original_feats.consumers)) <= 1:
            # If producer doesn't have other consumers -> remove original op
            assert prod_original_feats.consumers and prod_original_feats.consumers[0][0] == self.consumer_tag, \
                'Consumer not found in producer features'
            del bench_feats.operations[self.producer_tag]
            bench_feats.operation_tags.remove(self.producer_tag)
            # Remove the original op as a consumer from its producers
            for prod_producer in prod_producers:
                prod_prod_feats = bench_feats.operations[prod_producer]
                prod_prod_feats.consumers = [
                    (c, i) for c, i in prod_prod_feats.consumers
                    if c != self.producer_tag
                ]
        else:
            # Else -> remove this consumer from the original op
            prod_original_feats.consumers = [
                (c, i) for c, i in prod_original_feats.consumers
                if c != self.consumer_tag
            ]

    def __update_consumers_and_producers(self, prod_feats: OperationFeatures, state: OperationState):
        # Fused producer has only this op as a consumer
        # NOTE: To change with mutliple producers support
        # The consumption should be for one result only
        # NOTE: To change with mutliple uses support
        consumer_in_prod = [(c, i) for c, i in prod_feats.consumers if c == self.consumer_tag]
        prod_in_consumer = [(p, i) for p, i in state.operation_features.producers if p == self.producer_tag]
        consumed_results = [
            res_i for (_, res_i), (_, operand_i) in
            zip(consumer_in_prod, prod_in_consumer) if operand_i == self.producer_operand_idx
        ]
        consumed_result = consumed_results[0]
        if len(consumed_results) > 1:
            print_alert(
                "Having multiple used producer results isn't currently supported\n"
                f"Found [{consumed_results}], considering only first use {consumed_result}"
            )
        prod_feats.consumers = [(self.consumer_tag, consumed_result)]

        # Fused producer doesn't consume any other operation
        # NOTE: To change with mutliple producers support
        prod_feats.producers = []

    def __record_implicit_tiling(self, prod_feats: OperationFeatures, state: OperationState):
        consumer_feats = state.operation_features

        # 1. Get producer result tiling sizes
        prod_load = (consumer_feats.load_data + consumer_feats.store_data)[self.producer_operand_idx]
        consumer_args_tile_sizes = {
            nl.arg: self.parallel_params[i] if self.parallel_params[i] else nl.upper_bound
            for i, nl in enumerate(consumer_feats.nested_loops)
        }
        prod_res_tile_sizes: list[int] = []
        for dim_pos, dim_str in enumerate(prod_load):
            dim_str = dim_str.strip()
            dim_new_terms: list[str] = []
            for term in dim_str.split(' '):
                if term not in consumer_args_tile_sizes:
                    dim_new_terms.append(term)
                    continue
                # We need the last index not the size
                dim_new_terms.append(str(consumer_args_tile_sizes[term] - 1))
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
