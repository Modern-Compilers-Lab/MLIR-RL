func.func @conv(
    %1: tensor<128x14x14x864xf32>, %3: tensor<64x3x3x864xf32>, %7: tensor<128x12x12x64xf32>) -> tensor<128x12x12x64xf32>
  // This requests a C-compatible interface to be emitted for the function
  // when translating to LLVM IR.
  attributes { llvm.emit_c_interface }
{
  %D = linalg.conv_2d_nhwc_fhwc {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%1, %3 : tensor<128x14x14x864xf32>, tensor<64x3x3x864xf32>) outs(%7 : tensor<128x12x12x64xf32>) -> tensor<128x12x12x64xf32>
  
  return %D : tensor<128x12x12x64xf32> 
}

// Module containing the transformation script to be applied. The attribute
// is required to correctly verify the use of named (macro-like) sequences.
module attributes { transform.with_named_sequence } {

  transform.sequence failures(propagate) {
  // This argument will point to the top-level module.
  ^bb0(%arg0: !transform.any_op):

    

   %conv = transform.structured.match ops{["linalg.conv_2d_nhwc_fhwc"]} in %arg0
    : (!transform.any_op) -> !transform.any_op

    %generic_conv = transform.structured.generalize %conv : (!transform.any_op) -> !transform.any_op

    // %generic_conv2, %co = transform.structured.tile_using_forall %generic_conv tile_sizes [64, 6, 6, 8, 0, 0, 864] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    // %generic_conv2, %co = transform.structured.tile_using_forall %generic_conv tile_sizes [64, 6, 6, 8, 0, 0, 864] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


    %1, %2, %3, %loop = transform.structured.tile_reduction_using_for  %generic_conv by tile_sizes=[0, 0, 0, 0, 1, 1, 1] : (!transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op,
          !transform.any_op)


    
    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %f00 {
    } : !transform.any_op




    transform.include @lower failures(propagate) (%arg0)
      : (!transform.any_op) -> ()
    transform.yield
  }

  // Named sequence of transformations is a macro-like object that can be
  // included from another place in the transform dialect, but doesn't allow for
  // recursion. This can be reused in other scenarios.
  transform.named_sequence @lower(
      %arg0: !transform.any_op {transform.consumed}) {
    %f00 = transform.structured.match ops{["func.func"]} in %arg0
      : (!transform.any_op) -> !transform.any_op

    // Simplify the code as tiling and fusion may have produced a lot of
    // operations computing tensor subsets and loop ranges, some of which may be
    // duplicated or excessively complex. Simplification involving
    // canonicalization, common subexpression elimination, loop invariant code
    // motion and various rewrite patterns can be applied directly from the
    // transform dialect. Furthermore, an arbitrary combination of rewrite
    // patterns can be applied in one sweep to a given scope, a functionality
    // that cannot be achieved with conventional compiler passes that apply each
    // group of patterns separately (at least without creating a new pass for
    // each combination of pattern groups).
    transform.apply_patterns to %f00 {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.linalg.tiling_canonicalization
    } : !transform.any_op
    transform.apply_cse to %f00 : !transform.any_op
    %all_loops = transform.structured.match interface{LoopLikeInterface}
      in %arg0
      : (!transform.any_op) -> !transform.any_op
    transform.apply_licm to %all_loops : !transform.any_op

    // Tiling-by-one as a way of materializing loops produced operations
    // processing 4+D types where only a handful of dimension isn’t unit-sized,
    // e.g., tensor<1x1x1x5x64xf32> where 5 and 64 are tile sizes. Remove such
    // unit dimensions before vectorization, for clarity.
    transform.apply_patterns to %f00 {
      transform.apply_patterns.linalg.fold_unit_extent_dims_via_reshapes
    } : !transform.any_op

    // Vectorize the remaining non-unit dimensions in structured operations.
    // This essentially rewrites operations on `tensor<5x64xf32>` into
    // opreations on `vector<5x64xf32>`. Further lowering in MLIR and LLVM will
    // decompose this into a sequence of operations on single-dimensional
    // vectors of the platform-relevant size, e.g., `vector<16xf32>` for AVX512.
    // High-level vector primitives, such as `vector.transpose` and
    // `vector.broadcast` can be introduced at this stage. They will be later
    // lowered to sequences of lower-level primitives such as `vector.shuffle`
    // depending on the selected lowering strategy.
    %fv = transform.structured.vectorize_children_and_apply_patterns %f00
      : (!transform.any_op) -> !transform.any_op

    // Vectorization may have created new opportunities for cleanups. In
    // particular, tensor subsetting operations can be composed with vector
    // operations, and vector transfer (multi-dimensional load/store) operations
    // can be recombined and hoisted out of loops.
    transform.apply_patterns to %fv {
      transform.apply_patterns.canonicalization
      transform.apply_patterns.tensor.fold_tensor_subset_ops_into_vector_transfers
    } : !transform.any_op
    transform.apply_cse to %fv : !transform.any_op
    transform.structured.hoist_redundant_vector_transfers %fv
      : (!transform.any_op) -> !transform.any_op

    // Apply bufferization that rewrites the remaining operations on tensors
    // as operations on structured buffer (memref) types, including the function
    // API. MLIR bufferization uses destination-passing style meaning that a
    // buffer is shared between one of the operation's operands and its result.
    //
    // Since bufferization rewrites function signatures, it is applied as a
    // module-wise transformation. Therefore, it invalidates all previously
    // defined handles. Bufferization is usually a late step in the
    // transformation process, so invalidation is not an issue. However, if
    // other transformations, such as loop unrolling, are required after
    // bufferization, new handles should be produced using the match operations.
    //
    // One-shot bufferization itself does not produce buffer deallocations,
    // which may lead to leaks. So we have to run the buffer deallocation pass
    // pipeline to avoid them. Note that the transform dialect seamlessly runs
    // named passes and pass pipelines: if desired, one could replace complex
    // --pass-pipeline expressions with operations. Note that we apply the
    // pipeline to functions rather than entire module to avoid running it
    // on the transform IR that is contained in the module.
    %arg1 = transform.bufferization.one_shot_bufferize %arg0 {
      bufferize_function_boundaries = true,
      function_boundary_type_conversion = 1 : i32 }
      : (!transform.any_op) -> !transform.any_op
    %f = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_registered_pass "buffer-deallocation-pipeline" to %f
      : (!transform.any_op) -> !transform.any_op

    // Apply general canonicalization and CSE to each function after
    // bufferization as new simplification opportunities may have appeared.
    %fb = transform.structured.match ops{["func.func"]} in %arg1
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

    // // Lower complex, multidimensional vector operations into simpler
    // // primitives. This particular selection of the pattern groups corresponds
    // // to vector dialect operations present in the payload IR at this stage.
    // // Many of these groups can be parameterized to use different strategies or
    // // lower-level primitives offering performance trade-offs. In this case, we
    // // are selecting the simplest strategies.
    transform.apply_patterns to %fb {
      transform.apply_patterns.vector.lower_contraction
        lowering_strategy = parallelarith
      transform.apply_patterns.vector.lower_transfer
        max_transfer_rank = 1
      transform.apply_patterns.vector.lower_transpose
        lowering_strategy = eltwise
      transform.apply_patterns.vector.lower_shape_cast
    } : !transform.any_op

    // // These patterns apply in a separate sweep to avoid transfer-to-scf
    // // patterns overlap with lower-transfer patterns as they apply to the same
    // // kind of operations. These patterns may produce local allocations to act
    // // as temporary caches deep inside loops, which could lead to catastrophic
    // // performance. Such allocations are moved onto the stack and hoisted from
    // // all the surrounding loops.
    // transform.apply_patterns to %fb {
    //   transform.apply_patterns.vector.transfer_to_scf
    //   transform.apply_patterns.memref.alloc_to_alloca
    //   } : !transform.any_op
    // transform.bufferization.buffer_loop_hoisting %fb : !transform.any_op

    // // A final round of cleanups additionally includes patterns to simplify
    // // buffer aliasing operations that may have been introduced during
    // // bufferization and could result in excessively complex address
    // // computation.
    // transform.apply_patterns to %fb {
    //   transform.apply_patterns.memref.fold_memref_alias_ops
    //   transform.apply_patterns.canonicalization
    // } : !transform.any_op
    // transform.apply_cse to %fb : !transform.any_op

    transform.yield
  }
}

// The core computation, at the LLVM dialect level, must correspond to five
// immediately adjacent fma on vector<64xf32>.

// CHECK:      %[[R0:.+]] = llvm.mlir.undef : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[LINE0:.+]] = llvm.extractvalue %[[V:.+]][0] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA0:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE0]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R1:.+]] = llvm.insertvalue %[[FMA0]], %[[R0]][0]

// CHECK-NEXT: %[[LINE1:.+]] = llvm.extractvalue %[[V:.+]][1] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA1:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE1]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R2:.+]] = llvm.insertvalue %[[FMA1]], %[[R1]][1]

// CHECK-NEXT: %[[LINE2:.+]] = llvm.extractvalue %[[V:.+]][2] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA2:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE2]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R3:.+]] = llvm.insertvalue %[[FMA2]], %[[R2]][2]

// CHECK-NEXT: %[[LINE3:.+]] = llvm.extractvalue %[[V:.+]][3] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA3:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE3]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R4:.+]] = llvm.insertvalue %[[FMA3]], %[[R3]][3]

// CHECK-NEXT: %[[LINE4:.+]] = llvm.extractvalue %[[V:.+]][4] : !llvm.array<5 x vector<64xf32>>
// CHECK-NEXT: %[[FMA4:.+]] = llvm.intr.fma(%{{.*}}, %{{.*}}, %[[LINE4]])
// CHECK-SAME: -> vector<64xf32>
// CHECK-NEXT: %[[R5:.+]] = llvm.insertvalue %[[FMA4]], %[[R4]][4]
