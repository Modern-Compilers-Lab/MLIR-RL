// `tile_reduction_using_for` breaker: a non-associative reduction (division)
// whose accumulation chain cannot be split into partials.
//
// Sibling of tile_reduction_using_for_subtract_dep.mlir with a different
// non-reorderable combiner. The reduction is a running quotient:
// out[d0] = init[d0] / prod_d1 in[d0, d1], i.e. acc <- acc / a. Division is
// non-associative, so splitting the reduction into per-tile partials and
// merging them does not reproduce the sequential quotient — the accumulation
// order carries a real dependence that `tile_reduction_using_for` would break.
//
// As with subtraction, division has no reduction identity element, so the
// transform legally REFUSES to apply ("Failed to get an identity value for the
// reduction operation") and the schedule fails. The harness records the failed
// transform as outputs that differ and the MLIR detector as having flagged it.
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
        %s = arith.divf %acc, %a : f64
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
