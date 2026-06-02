// `transform.structured.fuse` into a tiled matmul, with the reduction tile loop
// interchanged outward (Category A — safe by construction, FP caveat).
//
// A producer adds 1 to A (A' = A + 1) and a matmul consumes it (C += A'·B).
// `fuse` tiles the matmul over all three dims [4, 4, 4] and interchanges them to
// [2, 0, 1], placing the K reduction tile loop outermost, then fuses the
// producer so each tile recomputes the A'-slice it needs.
//
// Tile-and-fuse recomputes the exact consumed producer slice via def-use, and
// tiling a linalg op covers the same iteration space; moving the reduction tile
// loop outward keeps each C[i, j] accumulated over k in index order (the
// sequential `scf.for` still threads the running sum), so the result is
// preserved exactly in integer arithmetic and within tolerance in floating point
// (rtol=1e-5/atol=1e-6). Outputs MATCH.
//
// `%C` is the destination-passing-style accumulator; after bufferization it
// becomes the in-place memref the harness compares. Both variants start from the
// same random C.
//
// Expected ground truth: outputs MATCH (within fp tolerance).
// Expected detectors: MLIR silent, Equivalence silent.
// Harness verdict: [PASS].

#id = affine_map<(d0, d1) -> (d0, d1)>

func.func @main(%A: tensor<8x8xf64>, %B: tensor<8x8xf64>, %C: tensor<8x8xf64>) -> tensor<8x8xf64> {
    %e = tensor.empty() : tensor<8x8xf64>
    %P = linalg.generic {tag = "producer",
        indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]
    } ins(%A : tensor<8x8xf64>) outs(%e : tensor<8x8xf64>) {
    ^bb0(%a: f64, %o: f64):
        %c1 = arith.constant 1.0 : f64
        %m = arith.addf %a, %c1 : f64
        linalg.yield %m : f64
    } -> tensor<8x8xf64>

    %0 = linalg.matmul {tag = "consumer"}
        ins(%P, %B : tensor<8x8xf64>, tensor<8x8xf64>)
        outs(%C : tensor<8x8xf64>) -> tensor<8x8xf64>

    return %0 : tensor<8x8xf64>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %cons = transform.structured.match attributes {tag = "consumer"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %fused, %loops:3 = transform.structured.fuse %cons [4, 4, 4] interchange [2, 0, 1]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
