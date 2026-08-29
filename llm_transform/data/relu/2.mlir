module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<256x512xf64>) -> (tensor<256x512xf64>, i64) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<256x512xf64>
    %arg1 = linalg.fill ins(%c0 : f64) outs(%new : tensor<256x512xf64>) -> tensor<256x512xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.generic {tag = "operation", indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<256x512xf64>) outs(%arg1 : tensor<256x512xf64>) {
    ^bb0(%in: f64, %out: f64):
      %zero = arith.constant 0.0 : f64
      %r = arith.maximumf %in, %zero : f64
      linalg.yield %r : f64
    } -> tensor<256x512xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<256x512xf64>, i64
  }
}
