// `transform.structured.fuse` (Category A — safe by construction).
//
// Tile-and-fuse a producer/consumer chain. A producer scales A by 2
// (P = A * 2), and a consumer adds B (C = P + B). `fuse` targets the consumer,
// tiles its 2D parallel iteration space [8, 8], and fuses the producer: the
// exact P-slice each consumer tile needs is recomputed inside the tile loop via
// the def-use chain.
//
// Tile-and-fuse is correct by construction: it operates on the structured ops'
// value-semantic (tensor) definitions and recomputes the consumed slice, so the
// fused, tiled program computes the same out = A*2 + B as the baseline. No
// iteration order of dependent computation is changed, so outputs must MATCH.
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
        %cons = transform.structured.match attributes {tag = "consumer"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %fused, %loops:2 = transform.structured.fuse %cons [8, 8]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
