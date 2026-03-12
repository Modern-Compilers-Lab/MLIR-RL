module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: memref<256x512xf64>, %arg1: memref<512x1024xf64>, %arg2: memref<256x1024xf64>) -> i64 attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    linalg.matmul {tag = "operation_0"} ins(%arg0, %arg1 : memref<256x512xf64>, memref<512x1024xf64>) outs(%arg2 : memref<256x1024xf64>)
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %3 : i64
  }
}
