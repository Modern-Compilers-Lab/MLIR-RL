// `tile_reduction_using_for` breaker: a non-associative, non-commutative
// reduction whose accumulation order cannot be split.
//
// This is the reduction-restructuring analogue of the tiling_*_dep breakers.
// Where tiling_col_dep.mlir reorders a spatial loop-carried dependence carried
// through aliasing memrefs, `tile_reduction_using_for` reorders the *reduction
// accumulation chain*: it strip-mines the reduction dimension into tiles,
// computes an identity-initialized partial per tile, then merges the partials.
// That restructuring is sound only when the combiner is associative AND
// commutative (the "safe by construction" guarantee of the Category-A
// classification).
//
// Here the combiner is subtraction: out[d0] = init[d0] - sum_d1 in[d0, d1],
// i.e. acc <- acc - a. Subtraction is neither associative nor commutative, so
// the per-tile-partials-then-merge form computes init + sum instead of
// init - sum — a genuine dependence violation of the accumulation order.
//
// The transform legally REFUSES to apply: subtraction has no reduction identity
// element, so MLIR raises "Failed to get an identity value for the reduction
// operation" and the schedule fails. The harness records the failed transform
// as outputs that differ and the MLIR detector as having flagged it.
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
        %s = arith.subf %acc, %a : f64
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
