// `transform.structured.split_reduction` (Category A, FP caveat).
//
// Splits the matmul's K reduction into a parallel + a reduction part (partial
// sums then a final combine). This is a mathematically valid reduction
// reordering: equal in exact arithmetic, and within tolerance in floating point.
// The harness compares with rtol=1e-5/atol=1e-6, so it should still MATCH.
//
// This is the legal counterpart to parallel_reduction.mlir: a reduction
// dimension is restructured, but here the partials are correctly combined, so
// the result is preserved. It also illustrates the verifier's deliberate
// order-freedom: it passes legal reduction reordering for the same reason it
// cannot flag an illegal reduction race.
//
// Expected ground truth: outputs MATCH (within fp tolerance).
// Expected detectors: MLIR silent, Legality silent, Equivalence silent.
// Harness verdict: [PASS].

func.func @main(%A: tensor<8x8xf64>, %B: tensor<8x8xf64>) -> tensor<8x8xf64> {
  %c0 = arith.constant 0.000000e+00 : f64
  %e = tensor.empty() : tensor<8x8xf64>
  %C = linalg.fill ins(%c0 : f64) outs(%e : tensor<8x8xf64>) -> tensor<8x8xf64>
  %0 = linalg.matmul ins(%A, %B : tensor<8x8xf64>, tensor<8x8xf64>)
                     outs(%C : tensor<8x8xf64>) -> tensor<8x8xf64>
  return %0 : tensor<8x8xf64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %init, %fill, %split, %combine = transform.structured.split_reduction %0
        { split_factor = 4, insert_split_dimension = 2 }
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
