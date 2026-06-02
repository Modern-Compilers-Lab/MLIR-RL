// `transform.structured.fuse_into_containing_op` (Category A — safe by
// construction). The two-step "tile then fuse into the tiled loop" pattern.
//
// A producer scales A by 2 (P = A * 2), and a consumer adds B (C = P + B). The
// schedule first tiles the consumer with `tile_using_forall` (num_threads
// [4, 4]) — producing an `scf.forall` containing op — then fuses the producer
// into that forall. Fusion recomputes, inside each thread, exactly the P-slice
// the consumer tile reads (computed from the def-use chain).
//
// Recomputing a pure elementwise producer per tile is correct: every output
// element C[i, j] = A[i, j]*2 + B[i, j] is independent, so the per-tile
// recomputation reproduces the baseline exactly. Outputs MATCH and no detector
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

        %tiled, %forall = transform.structured.tile_using_forall %cons num_threads [4, 4]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

        %fused, %new = transform.structured.fuse_into_containing_op %prod into %forall
            : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

        transform.yield
    }
}
