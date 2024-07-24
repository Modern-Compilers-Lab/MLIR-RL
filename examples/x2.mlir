func.func private @printMemrefF32(tensor<*xf32>)
func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func private @printFlops(f64)
func.func private @printI64(i64)

//!TTa = tensor<3x2048x2048x3xf32>
//!TTb = tensor<3x3x3x1xf32>
//!TTc = tensor<3x2046x2046x1xf32>


!TTa = tensor<32x230x230x3xf32>
!TTb = tensor<7x7x3x64xf32>
!TTc = tensor<32x112x112x64xf32>
func.func @conv() -> !TTc {

  %t0 = func.call @nanoTime() : () -> (i64)
%val = arith.constant 2.00000e+00 : f32
%out = bufferization.alloc_tensor() : !TTa
%input = linalg.fill ins(%val : f32) outs(%out : !TTa) -> !TTa
%out1 = bufferization.alloc_tensor() : !TTb
%filter = linalg.fill ins(%val : f32) outs(%out1 : !TTb) -> !TTb
%out2 = bufferization.alloc_tensor() : !TTc
%output = linalg.fill ins(%val : f32) outs(%out2 : !TTc) -> !TTc


%dense_ret = linalg.conv_2d_nhwc_hwcf  {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} 
ins(%input, %filter : !TTa,!TTb) 
outs(%output : !TTc) -> !TTc

%t1 = func.call @nanoTime() : () -> (i64)
%delta = arith.subi %t1, %t0 : i64
%fp = arith.uitofp %delta : i64 to f64
func.call @printFlops(%fp) : (f64) -> ()
func.call @printI64(%delta) : (i64) -> ()

  //%unranked = tensor.cast %dense_ret : !TTc to tensor<*xf32>
  //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()

  // Free the resources

  return %dense_ret : !TTc 
}
func.func @main(){
  %c1 = arith.constant 1: index
  %c0 = arith.constant 0 : index
  %n = arith.constant 10: index
  scf.for %i = %c0 to %n step %c1 {
    %outputmain = func.call @conv() : () -> !TTc
  }
  //%unranked = tensor.cast %outputmain : !TTc to tensor<*xf32>
  //call @printMemrefF32(%unranked) : (tensor<*xf32>) -> ()


  return
}


module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) 
{
  // The original fill op which will be fused into the outer scf.forall created by
  // tiling the convolution.
  %original_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // TODO: Add a transform.structured.specialize that can match a few different ops
  // Then, this reduces to just a linalg.matmul and we can reuse existing strategies.
  %named_conv = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
   %conv_l1 , %forall_l1= transform.structured.tile_using_forall %named_conv tile_sizes [ 2, 8 ]
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)


  %conv_l2, %loops_l2:7 = transform.structured.tile_using_for %conv_l1 [2, 1, 14, 16, 1, 7, 3] 
   : (!transform.any_op) -> (  !transform.any_op,!transform.any_op,!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  
  transform.structured.fuse_into_containing_op %original_fill into %forall_l1
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)
  %decomposed = transform.structured.decompose %conv_l2: (!transform.any_op) -> !transform.any_op

// Step 3. Vectorize.
  // ======================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op
   : (!transform.any_op) -> !transform.any_op
  %func_0 = transform.structured.vectorize_children_and_apply_patterns %func {vectorize_padding}
    : (!transform.any_op) -> (!transform.any_op)
//transform.structured.vectorize %conv_l1 {vectorize_padding}
   // : !transform.any_op

  %func_01 = transform.structured.hoist_redundant_vector_transfers %func_0 :
    (!transform.any_op) -> (!transform.any_op)


  // Step 4. Vector backend
  // ======================================================
  %f = transform.structured.match ops{["func.func"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  transform.apply_patterns to %f {


    transform.apply_patterns.vector.lower_contraction lowering_strategy = "outerproduct"

    transform.apply_patterns.vector.transfer_permutation_patterns

    transform.apply_patterns.vector.lower_multi_reduction lowering_strategy = "innerparallel"

    transform.apply_patterns.vector.split_transfer_full_partial split_transfer_strategy = "vector-transfer"

    transform.apply_patterns.vector.transfer_to_scf max_transfer_rank = 1 full_unroll = true

    transform.apply_patterns.vector.lower_transfer max_transfer_rank = 1

    transform.apply_patterns.vector.lower_shape_cast

    transform.apply_patterns.vector.lower_transpose lowering_strategy = "shuffle_1d"

    transform.apply_patterns.canonicalization
  } : !transform.any_op
  transform.yield
}
}
