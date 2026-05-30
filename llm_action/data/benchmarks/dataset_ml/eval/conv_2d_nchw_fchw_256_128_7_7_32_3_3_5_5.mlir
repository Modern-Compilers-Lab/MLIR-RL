module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<256x128x7x7xf64>, %arg1: tensor<32x128x3x3xf64>, %arg2: tensor<256x32x5x5xf64>) -> (tensor<256x32x5x5xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>, tag = "operation_0"} ins(%arg0, %arg1 : tensor<256x128x7x7xf64>, tensor<32x128x3x3xf64>) outs(%arg2 : tensor<256x32x5x5xf64>) -> tensor<256x32x5x5xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<256x32x5x5xf64>, i64
  }
}
