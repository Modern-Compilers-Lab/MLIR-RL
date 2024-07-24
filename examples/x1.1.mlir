#map = affine_map<(d0) -> (d0 * 12)>
#map1 = affine_map<(d0) -> (d0 * 2)>
#map2 = affine_map<(d0) -> (d0 * 25)>
  module attributes {torch.debug_module_name = "Net"} {
    func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
    func.func private @printFlops(f64)
    func.func private @printI64(i64)
    func.func private @printNewline()
    func.func private @printMemrefF32(tensor<*xf32>)
    func.func @matmul() -> tensor<1200x1000xf32> {
      %cst = arith.constant 2.000000e+00 : f32
      %0 = bufferization.alloc_tensor() : tensor<1200x1500xf32>
      %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<1200x1500xf32>) -> tensor<1200x1500xf32>
      %2 = bufferization.alloc_tensor() : tensor<1500x1000xf32>
      %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<1500x1000xf32>) -> tensor<1500x1000xf32>
      %4 = bufferization.alloc_tensor() : tensor<1200x1000xf32>
      %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<1200x1000xf32>) -> tensor<1200x1000xf32>
      %6 = call @nanoTime() : () -> i64
      %7 = scf.forall (%arg0, %arg1, %arg2) in (100, 40, 750) shared_outs(%arg3 = %5) -> (tensor<1200x1000xf32>) {
        %10 = affine.apply #map(%arg0)
        %11 = affine.apply #map1(%arg2)
        %12 = affine.apply #map1(%arg2)
        %13 = affine.apply #map2(%arg1)
        %14 = affine.apply #map(%arg0)
        %15 = affine.apply #map2(%arg1)
        %extracted_slice = tensor.extract_slice %1[%10, %11] [12, 2] [1, 1] : tensor<1200x1500xf32> to tensor<12x2xf32>
        %extracted_slice_0 = tensor.extract_slice %3[%12, %13] [2, 25] [1, 1] : tensor<1500x1000xf32> to tensor<2x25xf32>
        %extracted_slice_1 = tensor.extract_slice %arg3[%14, %15] [12, 25] [1, 1] : tensor<1200x1000xf32> to tensor<12x25xf32>
        %16 = linalg.matmul {tag = "operation_3"} ins(%extracted_slice, %extracted_slice_0 : tensor<12x2xf32>, tensor<2x25xf32>) outs(%extracted_slice_1 : tensor<12x25xf32>) -> tensor<12x25xf32>
        %17 = affine.apply #map(%arg0)
        %18 = affine.apply #map2(%arg1)
        scf.forall.in_parallel {
          tensor.parallel_insert_slice %16 into %arg3[%17, %18] [12, 25] [1, 1] : tensor<12x25xf32> into tensor<1200x1000xf32>
        }
      }
      %8 = call @nanoTime() : () -> i64
      %9 = arith.subi %8, %6 : i64
      call @printI64(%9) : (i64) -> ()
      call @printNewline() : () -> ()
      return %7 : tensor<1200x1000xf32>
    }
    func.func @main() {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      scf.for %arg0 = %c0 to %c2 step %c1 {
        %0 = func.call @matmul() : () -> tensor<1200x1000xf32>
      }
      return
    }
  }
  



module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) 
{


  %conv_gen_2 = transform.structured.match attributes{tag = "operation_3"} in %variant_op : (!transform.any_op) -> !transform.any_op
  %forall_op = transform.get_parent_op %conv_gen_2: (!transform.any_op) -> !transform.any_op
  
  %original_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  transform.structured.fuse_into_containing_op %original_fill into %forall_op : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)




  // %conv_gen_2 = transform.structured.match attributes{tag = "operation_3"} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %conv_l1 , %forall_l1= transform.structured.tile_using_forall %conv_gen_2 tile_sizes [ 2, 200 ] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // %original_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  // transform.structured.fuse_into_containing_op %original_fill into %forall_l1 : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

  // %fb2 = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %2 = transform.structured.vectorize_children_and_apply_patterns %fb2  : (!transform.any_op) -> !transform.any_op

  // %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
  // %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {vectorize_padding} : (!transform.any_op) -> (!transform.any_op)

  // %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 : (!transform.any_op) -> (!transform.any_op)

  // %f = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op

  // transform.apply_patterns to %f {
  //   transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"
  //   transform.apply_patterns.vector.transfer_permutation_patterns
  //   transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"
  //   transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"
  //   transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true
  //   transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1
  //   transform.apply_patterns.vector.lower_shape_cast
  //   transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"
  //   transform.apply_patterns.canonicalization
  // } : !transform.any_op

  transform.yield
}
}


// /scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt examples/x1.1.mlir -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule
// /scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt examples/x1.1.mlir -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule | /scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt  -loop-invariant-code-motion -cse -canonicalize -cse -eliminate-empty-tensors -empty-tensor-to-alloc-tensor -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -buffer-deallocation -convert-linalg-to-loops  -scf-foreach-thread-lowering -convert-vector-to-scf -convert-scf-to-openmp -canonicalize -lower-affine -expand-strided-metadata -finalize-memref-to-llvm -convert-scf-to-cf -lower-affine -convert-arith-to-llvm -convert-openmp-to-llvm -convert-vector-to-llvm -convert-cf-to-llvm -convert-func-to-llvm -convert-math-to-llvm -reconcile-unrealized-casts