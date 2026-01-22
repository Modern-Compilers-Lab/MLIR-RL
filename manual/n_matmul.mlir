func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
func.func @main(%arg0: tensor<24576x768xf64>, %arg1: tensor<768x384xf64>, %arg2: tensor<24576x384xf64>) -> (tensor<24576x384xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.matmul {tag = "operation"} ins(%arg0, %arg1 : tensor<24576x768xf64>, tensor<768x384xf64>) outs(%arg2 : tensor<24576x384xf64>) -> tensor<24576x384xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<24576x384xf64>, i64
}

// MC = 12288 | 4096, MC_thread = 1024
// KC = 256
// NC = 96 | 64
// MR = 6 | 4
// NR = 8
// M
// N
// K

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%module: !transform.any_op {transform.consumed}) {
        %op_tag = transform.param.constant "operation" -> !transform.any_param
        %arg0 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op
        %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        %gen = transform.structured.match attributes {tag = "operation"} in %arg0 : (!transform.any_op) -> !transform.any_op

        // --- Parallel ---
        %gen_1, %forall = transform.structured.tile_using_forall %gen tile_sizes [512, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- First Tiling ---
        %gen_2, %loops = transform.structured.tile_using_for %gen_1 tile_sizes [0, 0, 128] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_3 = transform.structured.promote %gen_2 {operands_to_promote = [0]} : (!transform.any_op) -> !transform.any_op
        %copy = transform.structured.match ops{["linalg.copy"]} in %loops : (!transform.any_op) -> !transform.any_op
        transform.structured.linalg_copy_to_memref %copy : (!transform.any_op) -> !transform.any_op
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_4, %loops_1 = transform.structured.tile_using_for %gen_3 tile_sizes [0, 64, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %gen_5 = transform.structured.promote %gen_4 {operands_to_promote = [1]} : (!transform.any_op) -> !transform.any_op
        %copy_1 = transform.structured.match ops{["linalg.copy"]} in %loops_1 : (!transform.any_op) -> !transform.any_op
        transform.structured.linalg_copy_to_memref %copy_1 : (!transform.any_op) -> !transform.any_op
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Second Tiling ---
        %gen_6, %loops_2:2 = transform.structured.tile_using_for %gen_5 tile_sizes [4, 8, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Vectorization ---
        transform.structured.vectorize %gen_6 {vectorize_nd_extract} : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        transform.apply_patterns to %f {
            transform.apply_patterns.vector.reduction_to_contract
            transform.apply_patterns.vector.transfer_permutation_patterns
        } : !transform.any_op
        transform.apply_patterns to %f {
            // transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.alloc_to_alloca
        } : !transform.any_op

        transform.apply_patterns to %f {
            transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
            transform.apply_patterns.vector.transfer_permutation_patterns
            transform.apply_patterns.vector.lower_outerproduct
            transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
            transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "linalg-copy"
            transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
            transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
            transform.apply_patterns.vector.lower_shape_cast
            transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        transform.yield
    }
    transform.named_sequence @tile_canonicalize(%arg0: !transform.any_op {transform.readonly}, %loop: !transform.any_op {transform.readonly}) {
        transform.apply_patterns to %loop {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.include @linalg_canonicalize failures(propagate) (%arg0) : (!transform.any_op) -> ()
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
