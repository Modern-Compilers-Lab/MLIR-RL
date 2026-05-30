#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<128x192x56x56xf64>, %arg1: tensor<128x192x56x56xf64>) -> (tensor<128x192x56x56xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"], tag = "operation_0"} ins(%arg0 : tensor<128x192x56x56xf64>) outs(%arg1 : tensor<128x192x56x56xf64>) {
    ^bb0(%in: f64, %out: f64):
      %cst = arith.constant 0.000000e+00 : f64
      %4 = arith.cmpf ugt, %in, %cst : f64
      %5 = arith.select %4, %in, %cst : f64
      linalg.yield %5 : f64
    } -> tensor<128x192x56x56xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<128x192x56x56xf64>, i64
  }
}
