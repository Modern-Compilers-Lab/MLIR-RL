#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0) -> (d0 floordiv 112)>
#map2 = affine_map<(d0) -> (d0 mod 112)>
#map3 = affine_map<(d0) -> (d0 floordiv 21)>
#map4 = affine_map<(d0) -> (d0 mod 21)>
#map5 = affine_map<(d0) -> ((d0 mod 21) floordiv 3)>
#map6 = affine_map<(d0) -> (d0 mod 3)>
#map7 = affine_map<(d0, d1) -> (d0 floordiv 112 + d1 floordiv 21)>
#map8 = affine_map<(d0, d1) -> (d0 mod 112 + (d1 mod 21) floordiv 3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func @conv() -> tensor<32x112x112x64xf32> {
    %0 = call @nanoTime() : () -> i64
    %cst = arith.constant 2.000000e+00 : f32
    %1 = bufferization.alloc_tensor() : tensor<32x230x230x3xf32>
    %2 = linalg.fill  ins(%cst : f32) outs(%1 : tensor<32x230x230x3xf32>) -> tensor<32x230x230x3xf32>
    %3 = bufferization.alloc_tensor() : tensor<7x7x3x64xf32>
    %4 = linalg.fill  ins(%cst : f32) outs(%3 : tensor<7x7x3x64xf32>) -> tensor<7x7x3x64xf32>
    %5 = bufferization.alloc_tensor() : tensor<32x112x112x64xf32>
    %6 = linalg.fill  ins(%cst : f32) outs(%5 : tensor<32x112x112x64xf32>) -> tensor<32x112x112x64xf32>
    %collapsed = tensor.collapse_shape %4 [[0, 1, 2], [3]] : tensor<7x7x3x64xf32> into tensor<147x64xf32>
    %collapsed_0 = tensor.collapse_shape %6 [[0], [1, 2], [3]] : tensor<32x112x112x64xf32> into tensor<32x12544x64xf32>
    %7 = tensor.empty() : tensor<32x12544x147xf32>
    %8 = linalg.generic {producerTag, indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%7 : tensor<32x12544x147xf32>) {
    ^bb0(%out: f32):
      %13 = linalg.index 0 : index
      %14 = linalg.index 1 : index
      %15 = linalg.index 2 : index
      %c112 = arith.constant 112 : index
      %c112_1 = arith.constant 112 : index
      %16 = affine.apply #map1(%14)
      %17 = affine.apply #map2(%14)
      %c7 = arith.constant 7 : index
      %c7_2 = arith.constant 7 : index
      %c3 = arith.constant 3 : index
      %c21 = arith.constant 21 : index
      %18 = affine.apply #map3(%15)
      %19 = affine.apply #map4(%15)
      %20 = affine.apply #map5(%15)
      %21 = affine.apply #map6(%15)
      %22 = affine.apply #map7(%14, %15)
      %23 = affine.apply #map8(%14, %15)
      %extracted = tensor.extract %2[%13, %22, %23, %21] : tensor<32x230x230x3xf32>
      linalg.yield %extracted : f32
    } -> tensor<32x12544x147xf32>
    %9 = linalg.generic {consumerTag,indexing_maps = [#map9, #map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%8, %collapsed : tensor<32x12544x147xf32>, tensor<147x64xf32>) outs(%collapsed_0 : tensor<32x12544x64xf32>) {
    ^bb0(%in: f32, %in_1: f32, %out: f32):
      %13 = arith.mulf %in, %in_1 : f32
      %14 = arith.addf %13, %out : f32
      linalg.yield %14 : f32
    } -> tensor<32x12544x64xf32>
    %expanded = tensor.expand_shape %9 [[0], [1, 2], [3]] : tensor<32x12544x64xf32> into tensor<32x112x112x64xf32>
    %10 = call @nanoTime() : () -> i64
    %11 = arith.subi %10, %0 : i64
    %12 = arith.uitofp %11 : i64 to f64
    call @printFlops(%12) : (f64) -> ()
    call @printI64(%11) : (i64) -> ()
    return %expanded : tensor<32x112x112x64xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c10 step %c1 {
      %0 = func.call @conv() : () -> tensor<32x112x112x64xf32>
    }
    return
  }
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%variant_op: !transform.any_op {transform.readonly}) 
{
%cons = transform.structured.match attributes{consumerTag} in %variant_op : (!transform.any_op) -> !transform.any_op

   %conv_l1 , %forall_l1= transform.structured.tile_using_forall %cons tile_sizes [ 4, 8, 32]
  : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

%prod = transform.structured.match attributes{producerTag} in %variant_op : (!transform.any_op) -> !transform.any_op 

  transform.structured.fuse_into_containing_op %prod into %forall_l1
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fb = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb : !transform.any_op

  %original_fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  transform.structured.fuse_into_containing_op %original_fill into %forall_l1
    : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    %fb1 = transform.structured.match ops{["func.func"]} in %variant_op
      : (!transform.any_op) -> !transform.any_op
    transform.apply_patterns to %fb1 {
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.apply_cse to %fb1 : !transform.any_op
  transform.yield
}
}



// /scratch/nb3891/Script/MLIR_RL_2/llvm-project/build/bin/mlir-opt examples/x3.mlir -transform-interpreter -canonicalize -test-transform-dialect-erase-schedule