module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(
    %arg0: memref<8x8x16x8x32xf64>,
    %arg1: memref<8x8x16x8x32xf64>
  ) -> i64 attributes {llvm.emit_c_interface} {
    %t0 = call @nanoTime() : () -> i64
    linalg.generic {
      indexing_maps = [
        affine_map<(a,b,c,d,e) -> (a,b,c,d,e)>,
        affine_map<(a,b,c,d,e) -> (a,b,c,d,e)>
      ],
      iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"],
      tag = "operation_0"
    } ins(%arg0 : memref<8x8x16x8x32xf64>)
      outs(%arg1 : memref<8x8x16x8x32xf64>) {
      ^bb0(%in: f64, %acc: f64):
        %sum = arith.addf %acc, %in : f64
        linalg.yield %sum : f64
    }
    %t1 = call @nanoTime() : () -> i64
    %dt = arith.subi %t1, %t0 : i64
    return %dt : i64
  }
}