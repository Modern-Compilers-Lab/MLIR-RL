module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<256x448x28x28xf64>, %arg1: tensor<448x448x1x1xf64>, %arg2: tensor<256x448x28x28xf64>) -> (tensor<256x448x28x28xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_0"} ins(%arg0, %arg1 : tensor<256x448x28x28xf64>, tensor<448x448x1x1xf64>) outs(%arg2 : tensor<256x448x28x28xf64>) -> tensor<256x448x28x28xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<256x448x28x28xf64>, i64
  }
}
