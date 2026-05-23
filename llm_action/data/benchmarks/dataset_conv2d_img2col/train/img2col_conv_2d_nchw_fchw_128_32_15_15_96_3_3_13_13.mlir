#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<()[s0] -> (s0 floordiv 9)>
#map2 = affine_map<()[s0] -> (s0 mod 9)>
#map3 = affine_map<()[s0] -> ((s0 mod 9) floordiv 3)>
#map4 = affine_map<()[s0] -> (s0 mod 3)>
#map5 = affine_map<()[s0] -> (s0 floordiv 13)>
#map6 = affine_map<()[s0] -> (s0 mod 13)>
#map7 = affine_map<()[s0, s1] -> (s0 floordiv 13 + (s1 mod 9) floordiv 3)>
#map8 = affine_map<()[s0, s1] -> (s0 + s1 - (s0 floordiv 13) * 13 - (s1 floordiv 3) * 3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<128x32x15x15xf64>, %arg1: tensor<96x32x3x3xf64>, %arg2: tensor<128x96x13x13xf64>) -> (tensor<128x96x13x13xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<96x32x3x3xf64> into tensor<96x288xf64>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<128x96x13x13xf64> into tensor<128x96x169xf64>
    %1 = tensor.empty() : tensor<128x288x169xf64>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%1 : tensor<128x288x169xf64>) {
    ^bb0(%out: f64):
      %6 = linalg.index 0 : index
      %7 = linalg.index 1 : index
      %8 = linalg.index 2 : index
      %c32 = arith.constant 32 : index
      %c3 = arith.constant 3 : index
      %c3_1 = arith.constant 3 : index
      %c3_2 = arith.constant 3 : index
      %c9 = arith.constant 9 : index
      %9 = affine.apply #map1()[%7]
      %10 = affine.apply #map2()[%7]
      %11 = affine.apply #map3()[%7]
      %12 = affine.apply #map4()[%7]
      %c13 = arith.constant 13 : index
      %c13_3 = arith.constant 13 : index
      %c13_4 = arith.constant 13 : index
      %13 = affine.apply #map5()[%8]
      %14 = affine.apply #map6()[%8]
      %15 = affine.apply #map7()[%8, %7]
      %16 = affine.apply #map8()[%8, %7]
      %extracted = tensor.extract %arg0[%6, %9, %15, %16] : tensor<128x32x15x15xf64>
      linalg.yield %extracted : f64
    } -> tensor<128x288x169xf64>
    %3 = linalg.generic {indexing_maps = [#map9, #map10, #map11], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %2 : tensor<96x288xf64>, tensor<128x288x169xf64>) outs(%collapsed_0 : tensor<128x96x169xf64>) attrs =  {tag = "operation_0"} {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %6 = arith.mulf %in, %in_1 : f64
      %7 = arith.addf %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<128x96x169xf64>
    %expanded = tensor.expand_shape %3 [[0], [1], [2, 3]] output_shape [128, 96, 13, 13] : tensor<128x96x169xf64> into tensor<128x96x13x13xf64>
    %4 = call @nanoTime() : () -> i64
    %5 = arith.subi %4, %0 : i64
    return %expanded, %5 : tensor<128x96x13x13xf64>, i64
  }
}
