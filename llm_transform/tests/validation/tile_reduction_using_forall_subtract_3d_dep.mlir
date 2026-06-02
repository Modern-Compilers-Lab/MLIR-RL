// `tile_reduction_using_forall` breaker, 3D shape: the same non-associative
// subtraction violation as tile_reduction_using_forall_subtract_dep.mlir, but on
// the parallel-parallel-reduction shape of tiling_elementwise_reduction_ok.mlir.
//
// The `scf.forall` counterpart of
// tile_reduction_using_for_subtract_3d_dep.mlir. Two parallel dims (d0, d1)
// carry an innermost reduction d2:
// out[d0, d1] = init[d0, d1] - sum_d2 in[d0, d1, d2], i.e. acc <- acc - a. Only
// the reduction dim d2 is distributed across threads (num_threads = [0, 0, 2]);
// the parallel dims are left intact. The violation is therefore purely in the
// reduction accumulation order, isolated from any spatial reordering.
//
// Subtraction is non-associative and non-commutative, and has no reduction
// identity element, so `tile_reduction_using_forall` legally REFUSES to split it
// ("Failed to get an identity value for the reduction operation") and the
// schedule fails. This variant confirms the rejection is independent of the
// surrounding parallel-iteration-space rank. The harness records the failed
// transform as outputs that differ and the MLIR detector as having flagged it.
//
// Expected ground truth: outputs DIFFER (transform rejected; no transformed
// kernel is produced).
// Expected detectors: MLIR detected (transform application failure).
// Harness verdict: [PASS].

func.func @main(%in: tensor<32x32x4xf64>, %out: tensor<32x32xf64>) -> tensor<32x32xf64> {
    %res = linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%in : tensor<32x32x4xf64>)
      outs(%out : tensor<32x32xf64>) {
    ^bb0(%a: f64, %acc: f64):
        %s = arith.subf %acc, %a : f64
        linalg.yield %s : f64
    } -> tensor<32x32xf64>
    return %res : tensor<32x32xf64>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %fill, %split, %combine, %forall =
            transform.structured.tile_reduction_using_forall %op by num_threads = [0, 0, 2]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
