// `transform.structured.pad` with a NON-NEUTRAL padding value.
// Category D1 (wrong values, not wrong order): every detector is SILENT, yet the
// result is wrong.
//
// matmul with K = 6. The schedule pads K up to a multiple of 4 (-> 8) but fills
// the two extra K columns of A and B with 1.0 instead of the neutral 0.0. Each
// padded term contributes 1.0 * 1.0 = 1.0 to the dot product, so every C[i][j]
// is too large by +2.
//
// Why nothing flags it:
//   * MLIR: the padding value is a trusted attribute; nothing is checked.
//   * Legality / Equivalence: the corruption flows through the *local* pad
//     buffer; the dependence structure on the function-argument buffers is
//     unchanged, so a dependence checker cannot see a data error.
//   Only the harness's numeric output comparison catches it.
//
// Expected ground truth: outputs DIFFER (off by +2 per element).
// Expected detectors: MLIR silent, Legality silent, Equivalence silent.
// Harness verdict: [FAIL] (illegal but undetected) — the intended demonstration
//   of a blind spot.

func.func @main(%A: tensor<4x6xf64>, %B: tensor<6x4xf64>) -> tensor<4x4xf64> {
  %c0 = arith.constant 0.000000e+00 : f64
  %e = tensor.empty() : tensor<4x4xf64>
  %C = linalg.fill ins(%c0 : f64) outs(%e : tensor<4x4xf64>) -> tensor<4x4xf64>
  %0 = linalg.matmul ins(%A, %B : tensor<4x6xf64>, tensor<6x4xf64>)
                     outs(%C : tensor<4x4xf64>) -> tensor<4x4xf64>
  return %0 : tensor<4x4xf64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %padded, %pad, %copy = transform.structured.pad %0
        pad_to_multiple_of [4, 4, 4] {
          padding_values = [1.000000e+00 : f64, 1.000000e+00 : f64, 1.000000e+00 : f64],
          padding_dimensions = [0, 1, 2],
          nofold_flags = [1, 1, 0],
          copy_back_op = "linalg.copy"
        } : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
