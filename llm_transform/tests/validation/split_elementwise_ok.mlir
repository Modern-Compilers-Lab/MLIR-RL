// transform.structured.split on a pure elementwise op: splitting the
// iteration domain into two complementary parts is order-preserving for an
// elementwise (no cross-iteration dependence) computation. Outputs MATCH.

func.func @main(%A: tensor<64x64xf64>) -> tensor<64x64xf64> {
  %e = tensor.empty() : tensor<64x64xf64>
  %0 = linalg.generic {
      indexing_maps = [
          affine_map<(d0, d1) -> (d0, d1)>,
          affine_map<(d0, d1) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel"]
  } ins(%A : tensor<64x64xf64>) outs(%e : tensor<64x64xf64>) {
  ^bb0(%in: f64, %out: f64):
      %c = arith.constant 2.0 : f64
      %m = arith.mulf %in, %c : f64
      linalg.yield %m : f64
  } -> tensor<64x64xf64>
  return %0 : tensor<64x64xf64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %op = transform.structured.match ops{["linalg.generic"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %split = transform.structured.split %op after 32 { dimension = 0 }
        : !transform.any_op
    transform.yield
  }
}
