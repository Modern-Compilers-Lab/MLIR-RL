module {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func @main(
    %arg0: tensor<[N]x[C]x[H]x[W]xf64>,
    %arg1: tensor<[F]x[C]x[KH]x[KW]xf64>,
    %arg2: tensor<[N]x[F]x[OH]x[OW]xf64>
  ) -> (tensor<[N]x[F]x[OH]x[OW]xf64>, i64)
  attributes {llvm.emit_c_interface} {
    %0 = call @nanoTime() : () -> i64
    %collapsed_filter = tensor.collapse_shape %arg1 [[0], [1, 2, 3]]
      : tensor<[F]x[C]x[KH]x[KW]xf64> into tensor<[F]x[C*KH*KW]xf64>
    %collapsed_output = tensor.collapse_shape %arg2 [[0], [1], [2, 3]]
      : tensor<[N]x[F]x[OH]x[OW]xf64> into tensor<[N]x[F]x[OH*OW]xf64>
    %img2col_buf = tensor.empty() : tensor<[N]x[C*KH*KW]x[OH*OW]xf64>
    %img2col = linalg.generic {
        indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
        iterator_types = ["parallel", "parallel", "parallel"]
    } outs(%img2col_buf : tensor<[N]x[C*KH*KW]x[OH*OW]xf64>) {
    ^bb0(%out: f64):
      %extracted = tensor.extract %arg0[...] : tensor<[N]x[C]x[H]x[W]xf64>
      linalg.yield %extracted : f64
    } -> tensor<[N]x[C*KH*KW]x[OH*OW]xf64>
    %result_flat = linalg.generic {
        indexing_maps = [
          affine_map<(d0, d1, d2, d3) -> (d1, d3)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>,
          affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
        ],
        iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%collapsed_filter, %img2col : tensor<[F]x[C*KH*KW]xf64>, tensor<[N]x[C*KH*KW]x[OH*OW]xf64>)
      outs(%collapsed_output : tensor<[N]x[F]x[OH*OW]xf64>)
      attrs = {tag = "operation_0"} {
    ^bb0(%in: f64, %in_1: f64, %out: f64):
      %mul = arith.mulf %in, %in_1 : f64
      %add = arith.addf %mul, %out : f64
      linalg.yield %add : f64
    } -> tensor<[N]x[F]x[OH*OW]xf64>
    %expanded = tensor.expand_shape %result_flat [[0], [1], [2, 3]]
      output_shape [[N], [F], [OH], [OW]]
      : tensor<[N]x[F]x[OH*OW]xf64> into tensor<[N]x[F]x[OH]x[OW]xf64>
    %1 = call @nanoTime() : () -> i64
    %2 = arith.subi %1, %0 : i64
    return %expanded, %2 : tensor<[N]x[F]x[OH]x[OW]xf64>, i64
  }
}