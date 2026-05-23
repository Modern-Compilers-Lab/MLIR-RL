#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<()[s0] -> (0)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<()[s0] -> (s0 floordiv 15)>
#map5 = affine_map<()[s0] -> (s0 mod 15)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<128x64x15x15xf64>, %arg1: tensor<96x64x1x1xf64>, %arg2: tensor<128x96x15x15xf64>) -> (tensor<128x96x15x15xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<96x64x1x1xf64> into tensor<96x64xf64>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<128x96x15x15xf64> into tensor<128x96x225xf64>
    %1 = tensor.empty() : tensor<128x64x225xf64>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%1 : tensor<128x64x225xf64>) {
    ^bb0(%out: f64):
      %6 = linalg.index 0 : index
      %7 = linalg.index 1 : index
      %8 = linalg.index 2 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %c1_2 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %9 = affine.apply #map1()[%7]
      %10 = affine.apply #map2()[%7]
      %11 = affine.apply #map3()
      %12 = affine.apply #map3()
      %c15 = arith.constant 15 : index
      %c15_4 = arith.constant 15 : index
      %c15_5 = arith.constant 15 : index
      %13 = affine.apply #map4()[%8]
      %14 = affine.apply #map5()[%8]
      %15 = affine.apply #map4()[%8]
      %16 = affine.apply #map5()[%8]
      %extracted = tensor.extract %arg0[%6, %9, %15, %16] : tensor<128x64x15x15xf64>
      linalg.yield %extracted : f64
    } -> tensor<128x64x225xf64>
    %3 = linalg.generic {indexing_maps = [#map6, #map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %2 : tensor<96x64xf64>, tensor<128x64x225xf64>) outs(%collapsed_0 : tensor<128x96x225xf64>) attrs =  {tag = "operation_0"} {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %6 = arith.mulf %in, %in_1 : f64
      %7 = arith.addf %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<128x96x225xf64>
    %expanded = tensor.expand_shape %3 [[0], [1], [2, 3]] output_shape [128, 96, 15, 15] : tensor<128x96x225xf64> into tensor<128x96x15x15xf64>
    %4 = call @nanoTime() : () -> i64
    %5 = arith.subi %4, %0 : i64
    return %expanded, %5 : tensor<128x96x15x15xf64>, i64
  }
}
