// `tile_reduction_using_forall` breaker: a non-associative, non-commutative
// reduction whose accumulation order cannot be distributed across threads.
//
// The `scf.forall` counterpart of tile_reduction_using_for_subtract_dep.mlir.
// `tile_reduction_using_forall` splits the reduction dimension across threads,
// each computing an identity-initialized partial, then merges the per-thread
// partials. That parallel restructuring is sound only when the combiner is
// associative AND commutative (the "safe by construction" guarantee of the
// Category-A classification).
//
// Here the combiner is subtraction: out[d0] = init[d0] - sum_d1 in[d0, d1],
// i.e. acc <- acc - a. Subtraction is neither associative nor commutative, so
// per-thread-partials-then-merge computes init + sum instead of init - sum — a
// genuine dependence violation of the accumulation order.
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

        %fill, %split, %combine, %forall =
            transform.structured.tile_reduction_using_forall %op by num_threads = [0, 8]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
