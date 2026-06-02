// `tile_reduction_using_for` breaker: applying the reduction-tiling strategy to
// a PARALLEL dimension instead of the reduction dimension.
//
// The kernel is a legal, ordinary sum reduction (out[d0] = sum_d1 in[d0, d1],
// acc <- acc + a) — the same body as tile_reduction_using_for_sum_ok.mlir. The
// illegality is entirely in the schedule: the tile-size vector [16, 0] requests
// tiling the parallel dimension d0 (and leaves the reduction d1 untiled).
//
// `tile_reduction_using_for` builds an identity-initialized partial and a merge
// for whichever dimension it tiles. Doing that to a parallel dimension would
// fold together iterations that produce *independent* output elements — every
// row d0 is a distinct result, not a value to be reduced — collapsing 64
// separate sums into one. This is the reduction-strategy counterpart of
// parallel_reduction.mlir's "treat a reduction as parallel": here we instead
// treat a parallel dim as a reduction.
//
// The transform legally REFUSES: MLIR raises "tiling parallel dimensions is not
// supported with partial reduction tiling strategies" and the schedule fails.
// The harness records the failed transform as outputs that differ and the MLIR
// detector as having flagged it.
//
// Expected ground truth: outputs DIFFER (transform rejected; no transformed
// kernel is produced).
// Expected detectors: MLIR detected (transform application failure).
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

        // Illegally tile the PARALLEL dim d0 (tile 16) and leave the reduction
        // d1 untiled (tile 0).
        %fill, %split, %combine, %for_op =
            transform.structured.tile_reduction_using_for %op by tile_sizes = [16, 0]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
