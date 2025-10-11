func.func private @nanoTime() -> i64 attributes { llvm.emit_c_interface }
func.func @main(%arg0: memref<256x128xf64>, %arg1: memref<128x256xf64>, %arg2: memref<256x256xf64>) -> i64 attributes { llvm.emit_c_interface } {
  %t0 = func.call @nanoTime() : () -> i64
  linalg.matmul ins(%arg0, %arg1 : memref<256x128xf64>, memref<128x256xf64>) outs(%arg2 : memref<256x256xf64>) 
  %t1 = func.call @nanoTime() : () -> i64
  %t2 = arith.subi %t1, %t0 : i64
  return %t2 : i64
}
