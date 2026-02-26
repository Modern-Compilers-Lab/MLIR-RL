// claude1159_3: For matmul_3 (256x512 × 512x1024, f64)
// Strategy: NC=8 instead of NC=16 (smaller B panels, more L1 friendly)
module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        %gen = transform.structured.match attributes {tag = "operation"} in %arg0 : (!transform.any_op) -> !transform.any_op

        %gen_1, %forall = transform.structured.tile_using_forall %gen tile_sizes [32, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_2, %loops = transform.structured.tile_using_for %gen_1 tile_sizes [0, 0, 256] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_3 = transform.structured.promote %gen_2 {operands_to_promote = [0], use_alloca, alignment = 64} : (!transform.any_op) -> !transform.any_op
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_4, %loops_1 = transform.structured.tile_using_for %gen_3 tile_sizes [0, 8, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_5 = transform.structured.promote %gen_4 {operands_to_promote = [1], use_alloca, alignment = 64} : (!transform.any_op) -> !transform.any_op
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_6, %loops_2:3 = transform.structured.tile_using_for %gen_5 tile_sizes [4, 8, 8] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        transform.structured.vectorize %gen_6 vector_sizes [4, 8, 8] : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        transform.apply_patterns to %f {
            transform.apply_patterns.vector.reduction_to_contract
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        transform.apply_patterns to %f {
            transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.vector.lower_outerproduct
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
            transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
            transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
            transform.apply_patterns.vector.lower_shape_cast
            transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        transform.yield
    }
    transform.named_sequence @linalg_canonicalize(%arg0: !transform.any_op {transform.readonly}) {
        %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
            transform.apply_patterns.memref.fold_memref_alias_ops
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()
        transform.yield
    }
    transform.named_sequence @licm(%arg0: !transform.any_op {transform.readonly}) {
        %all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_licm to %all_loops : !transform.any_op
        transform.yield
    }
}
