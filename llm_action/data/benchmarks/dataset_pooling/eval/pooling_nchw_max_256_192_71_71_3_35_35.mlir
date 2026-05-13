module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<256x192x71x71xf64>, %arg1: tensor<3x3xf64>, %arg2: tensor<256x192x35x35xf64>) -> (tensor<256x192x35x35xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.pooling_nchw_max {dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>, tag = "operation_0"} ins(%arg0, %arg1 : tensor<256x192x71x71xf64>, tensor<3x3xf64>) outs(%arg2 : tensor<256x192x35x35xf64>) -> tensor<256x192x35x35xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<256x192x35x35xf64>, i64
  }
}
