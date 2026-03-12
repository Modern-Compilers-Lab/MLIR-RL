module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: memref<24576x768xf64>, %arg1: memref<768x384xf64>, %arg2: memref<24576x384xf64>) -> i64 attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    linalg.matmul {tag = "operation_0"} ins(%arg0, %arg1 : memref<24576x768xf64>, memref<768x384xf64>) outs(%arg2 : memref<24576x384xf64>)
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %3 : i64
  }
}
