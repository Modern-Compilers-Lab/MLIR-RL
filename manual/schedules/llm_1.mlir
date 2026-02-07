// GEMM v978: Parallel outer tiling M=128, K cache=64
// Testing larger K cache

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%module_op: !transform.any_op) {
    %matmul = transform.structured.match attributes {tag = "operation"} in %module_op
      : (!transform.any_op) -> !transform.any_op

    // Parallel outer tile on M
    %tiled_outer, %forall = transform.structured.tile_using_forall %matmul tile_sizes [128, 0, 0]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // K cache tile = 64
    %tiled_cache, %loop_k_cache = transform.structured.tile_using_for %tiled_outer tile_sizes [0, 0, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    %promoted = transform.structured.promote %tiled_cache {
        operands_to_promote = [0, 1],
        use_full_tiles_by_default
    } : (!transform.any_op) -> !transform.any_op

    %tiled_micro, %loop_m, %loop_n, %loop_k = transform.structured.tile_using_for %promoted tile_sizes [4, 16, 1]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    %generic = transform.structured.generalize %tiled_micro : (!transform.any_op) -> !transform.any_op
    transform.structured.vectorize %generic vector_sizes [4, 16, 1] : !transform.any_op

    transform.loop.unroll %loop_k { factor = 8 } : !transform.any_op

    transform.loop.forall_to_parallel %forall : (!transform.any_op) -> !transform.any_op

    transform.apply_patterns to %module_op { transform.apply_patterns.vector.lower_masked_transfers } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.transfer_permutation_patterns } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.reduction_to_contract } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.lower_shape_cast } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d" } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.lower_broadcast } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.lower_outerproduct } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel" } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct" } : !transform.any_op
    transform.apply_patterns to %module_op { transform.apply_patterns.vector.transfer_to_scf } : !transform.any_op

    transform.yield
  }
}
