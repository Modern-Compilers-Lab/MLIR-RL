func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
func.func @main(%arg0: tensor<256x1536xf64>, %arg1: tensor<1536x1000xf64>) -> (tensor<256x1000xf64>, i64) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<256x1000xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<256x1000xf64>) -> tensor<256x1000xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.matmul {tag = "operation"} ins(%arg0, %arg1 : tensor<256x1536xf64>, tensor<1536x1000xf64>) outs(%arg2 : tensor<256x1000xf64>) -> tensor<256x1000xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<256x1000xf64>, i64
}
