// transform.structured.pack_transpose: pack a matmul, then transpose the
// layout of one packed operand. This is a pure layout change; the pack op
// and its consuming compute op are rewritten consistently, so the computed
// values are unchanged. Expect: outputs match -> level 0.

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
    %packed = transform.structured.pack %0 packed_sizes = [4, 4, 4]
        : (!transform.any_op) -> (!transform.any_op)
    // The pack feeding operand 0 (A) is the first linalg.pack.
    %packA = transform.get_producer_of_operand %packed[0]
        : (!transform.any_op) -> (!transform.op<"linalg.pack">)
    %packed_t, %pack_t, %unpack_t = transform.structured.pack_transpose %packA
        with_compute_op(%packed)
        inner_perm = [1, 0]
        : (!transform.op<"linalg.pack">, !transform.any_op)
          -> (!transform.any_op, !transform.any_op, !transform.any_op)
    // Lower all remaining pack/unpack ops so bufferization can proceed.
    %packs = transform.structured.match ops{["linalg.pack"]} in %arg0
        : (!transform.any_op) -> !transform.op<"linalg.pack">
    transform.structured.lower_pack %packs
        : (!transform.op<"linalg.pack">) -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
    %unpacks = transform.structured.match ops{["linalg.unpack"]} in %arg0
        : (!transform.any_op) -> !transform.op<"linalg.unpack">
    transform.structured.lower_unpack %unpacks
        : (!transform.op<"linalg.unpack">) -> (!transform.op<"tensor.empty">, !transform.op<"linalg.transpose">, !transform.op<"tensor.collapse_shape">, !transform.op<"tensor.extract_slice">)
    transform.yield
  }
}
