// Legal tile + interchange (Category A: safe).
//
// Format: kernel `@main` + an inner transform-schedule module, exactly like the
// files in tests/validation/. The harness (test_mlir_validation.py) runs the
// kernel with and without the schedule and compares the mutated buffers.
//
// matmul C = A*B tiled [4,4,4] with the M and N tile loops interchanged. Tiling
// and loop interchange of a linalg op are semantics-preserving by construction
// (linalg ops are interchange-invariant), so this is a legal reorder.
//
// Expected ground truth: outputs MATCH.
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
    %tiled, %l0, %l1, %l2 = transform.structured.tile_using_for %0
        tile_sizes [4, 4, 4] interchange = [1, 0, 2]
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
