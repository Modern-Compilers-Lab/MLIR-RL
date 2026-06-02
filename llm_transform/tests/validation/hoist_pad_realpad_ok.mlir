// transform.structured.hoist_pad where the pad actually grows the tile (K=12,
// tiled by [8,8,8] -> partial K tiles padded up to 8 with neutral 0.0). Hoisting
// the resulting tensor.pad out of a loop is value-preserving. Expect: outputs
// match -> level 0. Demonstrates hoist_pad is pure code motion of an existing
// (correctly-valued) pad and introduces no wrong values of its own.

func.func @main(%A: tensor<16x12xf64>, %B: tensor<12x16xf64>) -> tensor<16x16xf64> {
  %c0 = arith.constant 0.000000e+00 : f64
  %e = tensor.empty() : tensor<16x16xf64>
  %C = linalg.fill ins(%c0 : f64) outs(%e : tensor<16x16xf64>) -> tensor<16x16xf64>
  %0 = linalg.matmul ins(%A, %B : tensor<16x12xf64>, tensor<12x16xf64>)
                     outs(%C : tensor<16x16xf64>) -> tensor<16x16xf64>
  return %0 : tensor<16x16xf64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    %tiled, %l1, %l2, %l3 = transform.structured.tile_using_for %0 tile_sizes [8, 8, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    %padded, %pad, %copy = transform.structured.pad %tiled
        pad_to_multiple_of [1, 1, 1] {
          padding_values = [0.000000e+00 : f64, 0.000000e+00 : f64, 0.000000e+00 : f64],
          padding_dimensions = [0, 1, 2],
          nofold_flags = [1, 1, 1],
          copy_back_op = "linalg.copy"
        } : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)
    %padA = transform.get_producer_of_operand %padded[0]
        : (!transform.any_op) -> !transform.any_op
    %hoisted = transform.structured.hoist_pad %padA by 1 loops
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
