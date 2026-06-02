// transform.structured.hoist_pad: tile a matmul, pad the tiled op so a
// tensor.pad lives inside the loop nest, then hoist that pad out by 1 loop.
// Hoisting is value-preserving (the neutral 0.0 pad is used). Expect: applies
// and outputs match -> level 0 (or refuses to apply on this shape -> 0).

func.func @main(%A: tensor<16x16xf64>, %B: tensor<16x16xf64>) -> tensor<16x16xf64> {
  %c0 = arith.constant 0.000000e+00 : f64
  %e = tensor.empty() : tensor<16x16xf64>
  %C = linalg.fill ins(%c0 : f64) outs(%e : tensor<16x16xf64>) -> tensor<16x16xf64>
  %0 = linalg.matmul ins(%A, %B : tensor<16x16xf64>, tensor<16x16xf64>)
                     outs(%C : tensor<16x16xf64>) -> tensor<16x16xf64>
  return %0 : tensor<16x16xf64>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.matmul"]} in %arg0
        : (!transform.any_op) -> !transform.any_op
    // Tile so the matmul body sits inside scf.for loops.
    %tiled, %l1, %l2, %l3 = transform.structured.tile_using_for %0 tile_sizes [8, 8, 8]
        : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
    // Pad the tiled op; nofold so a real tensor.pad is produced inside the loop.
    %padded, %pad, %copy = transform.structured.pad %tiled
        pad_to_multiple_of [1, 1, 1] {
          padding_values = [0.000000e+00 : f64, 0.000000e+00 : f64, 0.000000e+00 : f64],
          padding_dimensions = [0, 1, 2],
          nofold_flags = [1, 1, 1],
          copy_back_op = "linalg.copy"
        } : (!transform.any_op)
            -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // Select operand A's single tensor.pad and hoist it out of 1 loop.
    %padA = transform.get_producer_of_operand %padded[0]
        : (!transform.any_op) -> !transform.any_op
    %hoisted = transform.structured.hoist_pad %padA by 1 loops
        : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
