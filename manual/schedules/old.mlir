// MC = 12288 | 4096, MC_thread = 1024
// KC = 256
// NC = 96 | 64
// MR = 6 | 4
// NR = 8
// M
// N
// K

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.consumed}) {
        %op_tag = transform.param.constant "operation" -> !transform.any_param
        %f = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
        %matmul = transform.structured.match attributes {tag = "operation"} in %arg0 : (!transform.any_op) -> !transform.any_op
        %gen = transform.structured.generalize %matmul : (!transform.any_op) -> !transform.any_op

        // --- Packing ---
        %gen_1 = transform.structured.pack %gen packed_sizes = [1024, 96, 256] : (!transform.any_op) -> !transform.any_op
        transform.annotate %gen_1 "tag" = %op_tag : !transform.any_op, !transform.any_param
        %pack = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.pack">
        %a:2, %pack_linalg = transform.structured.lower_pack %pack : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
        %unpack = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.unpack">
        %b, %unpack_linalg, %c:2 = transform.structured.lower_unpack %unpack : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op

        // --- Interchange ---
        %gen_2 = transform.structured.interchange %gen_1 iterator_interchange = [0, 2, 1, 3, 4, 5] : (!transform.any_op) -> !transform.any_op

        // --- Parallel ---
        %gen_3, %forall = transform.structured.tile_using_forall %gen_2 tile_sizes [1, 0, 0, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.apply_patterns to %forall {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Tiling ---
        %gen_5, %loops:2 = transform.structured.tile_using_for %gen_3 tile_sizes [1, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.apply_patterns to %loops {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Fusion (2) ---
        %pack_linalgs:3 = transform.split_handle %pack_linalg : (!transform.op<"linalg.transpose">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %transpose_fused, %containing_loop = transform.structured.fuse_into_containing_op %pack_linalgs#0 into %forall : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %transpose_fused_1, %containing_loop_1 = transform.structured.fuse_into_containing_op %pack_linalgs#1 into %containing_loop : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %transpose_fused_2, %containing_loop_2 = transform.structured.fuse_into_containing_op %pack_linalgs#2 into %containing_loop_1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.apply_patterns to %containing_loop_2 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Optimize Pack #2 ---
        %transpose_prll_1, %transpose_loops:2 = transform.structured.tile_using_for %transpose_fused_2 tile_sizes [0, 0, 16, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.apply_patterns to %transpose_loops {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %transpose_prll_2, %transpose_loops_1 = transform.structured.tile_using_for %transpose_prll_1 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.apply_patterns to %transpose_loops_1 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        transform.structured.vectorize %transpose_prll_2 {vectorize_nd_extract} : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Optimize Pack #0 ---
        %transpose_prll_3, %transpose_loops_2:2 = transform.structured.tile_using_for %transpose_fused tile_sizes [0, 0, 8, 4] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.apply_patterns to %transpose_loops_2 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        %transpose_prll_4, %transpose_loops_3 = transform.structured.tile_using_for %transpose_prll_3 tile_sizes [0, 0, 1, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.apply_patterns to %transpose_loops_3 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        transform.structured.vectorize %transpose_prll_4 {vectorize_nd_extract} : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Packing ---
        %gen_6 = transform.structured.pack %gen_5 packed_sizes = [4, 8, 0] : (!transform.any_op) -> !transform.any_op
        transform.annotate %gen_6 "tag" = %op_tag : !transform.any_op, !transform.any_param
        %pack_1 = transform.structured.match ops{["linalg.pack"]} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.pack">
        %a_1:2, %pack_linalg_1 = transform.structured.lower_pack %pack_1 : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
        %unpack_1 = transform.structured.match ops{["linalg.unpack"]} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.unpack">
        %b_1, %unpack_linalg_1, %c_1:2 = transform.structured.lower_unpack %unpack_1 : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op

        // --- Tiling ---
        %gen_7, %loops_2:2 = transform.structured.tile_using_for %gen_6 tile_sizes [1, 1, 0, 0, 0] : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        transform.apply_patterns to %loops_2#0 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Fusion (2) ---
        %pack_linalgs_1:3 = transform.split_handle %pack_linalg_1 : (!transform.op<"linalg.transpose">) -> (!transform.any_op, !transform.any_op, !transform.any_op)
        %transpose_fused_3, %containing_loop_3 = transform.structured.fuse_into_containing_op %pack_linalgs_1#0 into %loops_2#0 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        %transpose_fused_4, %containing_loop_4 = transform.structured.fuse_into_containing_op %pack_linalgs_1#1 into %containing_loop_3 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
        transform.apply_patterns to %containing_loop_4 {
            transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.linalg.tiling_canonicalization
        } : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        // --- Interchange ---
        %gen_8 = transform.structured.interchange %gen_7 iterator_interchange = [1, 2, 0] : (!transform.any_op) -> !transform.any_op

        // --- Vectorization ---
        transform.structured.vectorize %gen_8 {vectorize_nd_extract} : !transform.any_op
        transform.include @licm failures(propagate) (%arg0) : (!transform.any_op) -> ()

        transform.structured.eliminate_empty_tensors %arg0 : !transform.any_op
        %empty = transform.structured.match ops{["tensor.empty"]} in %arg0 : (!transform.any_op) -> !transform.op<"tensor.empty">
        transform.bufferization.empty_tensor_to_alloc_tensor %empty : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">

        transform.apply_patterns to %f {
            transform.apply_patterns.vector.reduction_to_contract
            transform.apply_patterns.vector.transfer_permutation_patterns
        } : !transform.any_op
        transform.apply_patterns to %f {
            transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
            transform.apply_patterns.canonicalization
        } : !transform.any_op

        %arg1 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %arg0 {bufferize_function_boundaries = true} : (!transform.any_op) -> !transform.any_op
        %f1 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        transform.apply_patterns to %f1 {
            transform.apply_patterns.canonicalization
            transform.apply_patterns.memref.alloc_to_alloca
        } : !transform.any_op

        transform.apply_patterns to %f1 {
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
    transform.named_sequence @licm(%arg0: !transform.any_op {transform.readonly}) {
        %all_loops = transform.structured.match interface{LoopLikeInterface} in %arg0 : (!transform.any_op) -> !transform.any_op
        transform.apply_licm to %all_loops : !transform.any_op
        transform.yield
    }
}
