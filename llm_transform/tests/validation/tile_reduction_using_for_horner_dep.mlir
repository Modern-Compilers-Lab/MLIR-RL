// `tile_reduction_using_for` breaker: a positional (Horner-style) accumulation
// whose loop-carried dependence is genuinely sequential.
//
// This is the closest reduction-domain analogue of the spatial tiling_*_dep
// breakers: every step feeds the previous accumulator through a multiply before
// adding the new element — acc <- acc * 2 + a — so the contribution of element
// in[d0, d1] is scaled by 2^(N-1-d1). The result is the Horner evaluation of a
// polynomial; the accumulation order is load-bearing and cannot be regrouped.
//
// `tile_reduction_using_for` assumes a single associative/commutative combiner
// that it can split into per-tile partials and merge. This body has no such
// combiner: the accumulator is consumed by a `mulf` and the new element by an
// `addf`, so the value carried across iterations is not a plain reduction.
// MLIR cannot recognize a reduction combiner and legally REFUSES to apply the
// transform ("Failed to anaysis the reduction operation"); the schedule fails.
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
        %c = arith.constant 2.0 : f64
        %t = arith.mulf %acc, %c : f64
        %s = arith.addf %t, %a : f64
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
