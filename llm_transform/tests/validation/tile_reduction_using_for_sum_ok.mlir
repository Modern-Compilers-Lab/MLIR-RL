// `transform.structured.tile_reduction_using_for` (Category A, FP caveat).
//
// The canonical reduction-tiling shape: a row-wise sum over an innermost
// reduction dimension, out[d0] = sum_d1 in[d0, d1]. `tile_reduction_using_for`
// splits the reduction dimension d1 into tiles, accumulates a partial sum per
// tile into an identity-initialized temporary, then runs a final `for`-carried
// merge that reduces the partials back into the original output.
//
// This is a mathematically valid reduction reordering (identity-init + merge):
// equal in exact arithmetic and within tolerance in floating point. The harness
// compares with rtol=1e-5/atol=1e-6, so the transformed kernel must MATCH the
// baseline.
//
// `%out` is the destination-passing-style accumulator; after bufferization it
// becomes the in-place memref the harness compares. The length-128 reduction is
// a multiple of the tile size 16, so only full tiles are produced.
//
// Expected ground truth: outputs MATCH (within fp tolerance).
// Expected detectors: MLIR silent, Equivalence silent.
// Harness verdict: [PASS].

func.func @main(%in: tensor<64x128xf64>, %out: tensor<64xf64>) -> tensor<64xf64> {
    %res = linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>
        ],
        iterator_types = ["parallel", "reduction"]
    } ins(%in : tensor<64x128xf64>)
      outs(%out : tensor<64xf64>) {
    ^bb0(%a: f64, %acc: f64):
        %s = arith.addf %a, %acc : f64
        linalg.yield %s : f64
    } -> tensor<64xf64>
    return %res : tensor<64xf64>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %fill, %split, %combine, %for_op =
            transform.structured.tile_reduction_using_for %op by tile_sizes = [0, 16]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
