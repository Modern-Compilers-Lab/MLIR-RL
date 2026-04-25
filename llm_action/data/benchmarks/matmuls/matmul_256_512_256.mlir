module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<256x512xf64>, %arg1: tensor<512x256xf64>, %arg2: tensor<256x256xf64>) -> (tensor<256x256xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.matmul ins(%arg0, %arg1 : tensor<256x512xf64>, tensor<512x256xf64>) outs(%arg2 : tensor<256x256xf64>) -> tensor<256x256xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<256x256xf64>, i64
  }
}
