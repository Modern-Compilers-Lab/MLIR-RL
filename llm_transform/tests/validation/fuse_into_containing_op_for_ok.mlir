// `transform.structured.fuse_into_containing_op` into a sequential `scf.for`
// (Category A — safe by construction). Sibling of
// fuse_into_containing_op_forall_ok.mlir with an `scf.for` containing op.
//
// Same producer/consumer chain (P = A * 2, C = P + B). The schedule tiles the
// consumer with `tile_using_for` (tile_sizes [8, 8]) — producing nested
// `scf.for` loops — then fuses the producer into the outer `scf.for`. Fusion
// recomputes the consumed P-slice inside the loop via the def-use chain.
//
// Recomputing a pure elementwise producer per tile is correct, so the fused,
// tiled program reproduces out = A*2 + B exactly. Outputs MATCH and no detector
// fires.
//
// `%out` is the destination-passing-style result; after bufferization it becomes
// the in-place memref the harness compares.
//
// Expected ground truth: outputs MATCH.
// Expected detectors: MLIR silent, Equivalence silent.
// Harness verdict: [PASS].

#id = affine_map<(d0, d1) -> (d0, d1)>

func.func @main(%A: tensor<64x64xf64>, %B: tensor<64x64xf64>, %out: tensor<64x64xf64>) -> tensor<64x64xf64> {
    %e = tensor.empty() : tensor<64x64xf64>
    %P = linalg.generic {tag = "producer",
        indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]
    } ins(%A : tensor<64x64xf64>) outs(%e : tensor<64x64xf64>) {
    ^bb0(%a: f64, %o: f64):
        %c = arith.constant 2.0 : f64
        %m = arith.mulf %a, %c : f64
        linalg.yield %m : f64
    } -> tensor<64x64xf64>

    %C = linalg.generic {tag = "consumer",
        indexing_maps = [#id, #id, #id], iterator_types = ["parallel", "parallel"]
    } ins(%P, %B : tensor<64x64xf64>, tensor<64x64xf64>) outs(%out : tensor<64x64xf64>) {
    ^bb0(%p: f64, %b: f64, %o: f64):
        %s = arith.addf %p, %b : f64
        linalg.yield %s : f64
    } -> tensor<64x64xf64>

    return %C : tensor<64x64xf64>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %prod = transform.structured.match attributes {tag = "producer"} in %arg0
            : (!transform.any_op) -> !transform.any_op
        %cons = transform.structured.match attributes {tag = "consumer"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %tiled, %l0, %l1 = transform.structured.tile_using_for %cons tile_sizes [8, 8]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        %fused, %new = transform.structured.fuse_into_containing_op %prod into %l0
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}
