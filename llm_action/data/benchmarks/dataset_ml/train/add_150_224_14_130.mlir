module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(%arg0: tensor<150x224x14x130xf64>, %arg1: tensor<150x224x14x130xf64>, %arg2: tensor<150x224x14x130xf64>) -> (tensor<150x224x14x130xf64>, i64) attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %1 = linalg.add {tag = "operation_0"} ins(%arg0, %arg1 : tensor<150x224x14x130xf64>, tensor<150x224x14x130xf64>) outs(%arg2 : tensor<150x224x14x130xf64>) -> tensor<150x224x14x130xf64>
    %2 = call @nanoTime() : () -> i64
    %3 = arith.subi %2, %0 : i64
    return %1, %3 : tensor<150x224x14x130xf64>, i64
  }
}
