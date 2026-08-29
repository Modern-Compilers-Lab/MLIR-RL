// Attempt to break transform.structured.split_reduction by using a large
// split factor and inner_parallel placement on a sizeable matmul, maximizing
// the floating-point reassociation. split_reduction only reorders an
// associative reduction (partial sums then final combine): equal in exact
// arithmetic, within tolerance in FP. Expected: outputs MATCH.

func.func @main(%A: tensor<16x256xf64>, %B: tensor<256x16xf64>) -> tensor<16x16xf64> {
  %c0 = arith.constant 0.000000e+00 : f64
  %e = tensor.empty() : tensor<16x16xf64>
  %C = linalg.fill ins(%c0 : f64) outs(%e : tensor<16x16xf64>) -> tensor<16x16xf64>
  %0 = linalg.matmul ins(%A, %B : tensor<16x256xf64>, tensor<256x16xf64>)
                     outs(%C : tensor<16x16xf64>) -> tensor<16x16xf64>
  return %0 : tensor<16x16xf64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %init, %fill, %split, %combine = transform.structured.split_reduction %0
        { split_factor = 32, insert_split_dimension = 2, inner_parallel }
        : (!transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    transform.yield
  }
}
