module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<128x128x14x14xf64>, %arg1: tensor<96x128x3x3xf64>) -> (tensor<128x96x6x6xf64>, i64) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<128x96x6x6xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<128x96x6x6xf64>) -> tensor<128x96x6x6xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.conv_2d_nchw_fchw {tag = "operation", dilations = dense<1> : tensor<2xi64>, strides = dense<2> : tensor<2xi64>} ins(%arg0, %arg1 : tensor<128x128x14x14xf64>, tensor<96x128x3x3xf64>) outs(%arg2 : tensor<128x96x6x6xf64>) -> tensor<128x96x6x6xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<128x96x6x6xf64>, i64
  }
}
