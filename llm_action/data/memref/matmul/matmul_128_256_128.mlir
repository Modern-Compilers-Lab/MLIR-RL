module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: memref<128x256xf64>, %arg1: memref<256x128xf64>, %arg2: memref<128x128xf64>) -> i64 attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    linalg.matmul {tag = "operation_0"} ins(%arg0, %arg1 : memref<128x256xf64>, memref<256x128xf64>) outs(%arg2 : memref<128x128xf64>)
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %3 : i64
  }
}
