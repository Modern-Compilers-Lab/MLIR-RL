#map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<()[s0] -> (s0)>
#map2 = affine_map<()[s0] -> (0)>
#map3 = affine_map<() -> (0)>
#map4 = affine_map<()[s0] -> (s0 floordiv 8)>
#map5 = affine_map<()[s0] -> (s0 mod 8)>
#map6 = affine_map<()[s0] -> ((s0 floordiv 8) * 2)>
#map7 = affine_map<()[s0] -> (s0 * 2 - (s0 floordiv 8) * 16)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<256x192x15x15xf64>, %arg1: tensor<32x192x1x1xf64>, %arg2: tensor<256x32x8x8xf64>) -> (tensor<256x32x8x8xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %collapsed = tensor.collapse_shape %arg1 [[0], [1, 2, 3]] : tensor<32x192x1x1xf64> into tensor<32x192xf64>
    %collapsed_0 = tensor.collapse_shape %arg2 [[0], [1], [2, 3]] : tensor<256x32x8x8xf64> into tensor<256x32x64xf64>
    %1 = tensor.empty() : tensor<256x192x64xf64>
    %2 = linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel", "parallel"]} outs(%1 : tensor<256x192x64xf64>) {
    ^bb0(%out: f64):
      %6 = linalg.index 0 : index
      %7 = linalg.index 1 : index
      %8 = linalg.index 2 : index
      %c192 = arith.constant 192 : index
      %c1 = arith.constant 1 : index
      %c1_1 = arith.constant 1 : index
      %c1_2 = arith.constant 1 : index
      %c1_3 = arith.constant 1 : index
      %9 = affine.apply #map1()[%7]
      %10 = affine.apply #map2()[%7]
      %11 = affine.apply #map3()
      %12 = affine.apply #map3()
      %c8 = arith.constant 8 : index
      %c8_4 = arith.constant 8 : index
      %c8_5 = arith.constant 8 : index
      %13 = affine.apply #map4()[%8]
      %14 = affine.apply #map5()[%8]
      %15 = affine.apply #map6()[%8]
      %16 = affine.apply #map7()[%8]
      %extracted = tensor.extract %arg0[%6, %9, %15, %16] : tensor<256x192x15x15xf64>
      linalg.yield %extracted : f64
    } -> tensor<256x192x64xf64>
    %3 = linalg.generic {indexing_maps = [#map8, #map9, #map10], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %2 : tensor<32x192xf64>, tensor<256x192x64xf64>) outs(%collapsed_0 : tensor<256x32x64xf64>) attrs =  {tag = "operation_0"} {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %6 = arith.mulf %in, %in_1 : f64
      %7 = arith.addf %6, %out : f64
      linalg.yield %7 : f64
    } -> tensor<256x32x64xf64>
    %expanded = tensor.expand_shape %3 [[0], [1], [2, 3]] output_shape [256, 32, 8, 8] : tensor<256x32x64xf64> into tensor<256x32x8x8xf64>
    %4 = call @nanoTime() : () -> i64
    %5 = arith.subi %4, %0 : i64
    return %expanded, %5 : tensor<256x32x8x8xf64>, i64
  }
}
