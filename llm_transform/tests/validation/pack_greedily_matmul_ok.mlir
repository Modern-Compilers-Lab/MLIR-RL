// transform.structured.pack_greedily on a matmul. The op infers the (m,n,k)
// matmul structure and packs with the given inner sizes. It is a pure layout
// change (pad amounts use the neutral value), so computed values are unchanged.
// Expect: outputs match -> level 0.

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
    %packed = transform.structured.pack_greedily %0
        matmul_packed_sizes = [4, 4, 4]
        matmul_inner_dims_order = [0, 1, 2]
        : (!transform.any_op) -> !transform.any_op
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
