func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
func.func @main(%arg0: tensor<24576x768xf64>, %arg1: tensor<768x384xf64>) -> (tensor<24576x384xf64>, i64) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0.0 : f64
    %new = tensor.empty() : tensor<24576x384xf64>
    %arg2 = linalg.fill ins(%c0 : f64) outs(%new : tensor<24576x384xf64>) -> tensor<24576x384xf64>
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.matmul {tag = "operation"} ins(%arg0, %arg1 : tensor<24576x768xf64>, tensor<768x384xf64>) outs(%arg2 : tensor<24576x384xf64>) -> tensor<24576x384xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<24576x384xf64>, i64
}
