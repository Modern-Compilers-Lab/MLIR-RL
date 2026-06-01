// Structurally invalid interchange (Category B: MLIR detects it).
//
// `interchange = [0, 0, 1]` is not a permutation of {0,1,2}. The transform
// interpreter rejects it while applying the schedule, so MLIR itself catches the
// error before any wrong code is produced. (Note: a *valid* permutation here
// would always be semantically legal — a single linalg op is interchange-
// invariant — which is exactly why the dangerous reorders live in the other
// files, not in plain `interchange`.)
//
// Expected ground truth: transform fails to apply (no executable transformed
//   kernel) -> harness reports [UNKW].
// Expected detectors: MLIR detected (the apply-time error); others n/a.

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
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %0
        tile_sizes [4, 4, 4] interchange = [0, 0, 1]
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
