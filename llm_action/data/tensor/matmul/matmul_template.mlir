module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(
    %arg0: tensor<[I]x[J]xf64>,
    %arg1: tensor<[J]x[K]xf64>,
    %arg2: tensor<[I]x[K]xf64>
    ) -> (tensor<[I]x[K]xf64>, i64) attributes {llvm.emit_c_interface} {
      %0 = call @nanoTime() : () -> i64
      %1 = linalg.matmul {tag = "operation_0"} ins(%arg0, %arg1 : tensor<[I]x[J]xf64>, tensor<[J]x[K]xf64>) outs(%arg2 : tensor<[I]x[K]xf64>) -> tensor<[I]x[K]xf64>
      %2 = call @nanoTime() : () -> i64
      %3 = arith.subi %2, %0 : i64
      return %1, %3 : tensor<[I]x[K]xf64>, i64
  }
}