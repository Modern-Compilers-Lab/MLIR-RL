// `transform.structured.fuse` with an interchange of the WRONG LENGTH for the
// op's iteration rank (Category B: MLIR detects it at apply time).
//
// The consumer has a 2-D iteration space (d0, d1), but the interchange attribute
// `[2, 1, 0]` has three entries. It IS a permutation of {0, 1, 2}, so it passes
// FuseOp's static permutation verifier — but index 2 (and the extra entry) is
// out of range for a 2-dimensional op. The mismatch is therefore not caught at
// parse/verify time; it is caught only when the transform interpreter *applies*
// the schedule and tries to permute loops that do not exist.
//
// This is the `fuse` counterpart of interchange_nonpermutation.mlir, expressed
// through length/rank mismatch rather than a repeated index: a same-rank
// repeated-index interchange (e.g. `[1, 1]`) is instead rejected by the static
// verifier at parse time, which the harness cannot route through its graceful
// failed-transform path. A wrong-length permutation slips past the static
// verifier and fails during application, so the harness sees a clean
// failed-to-apply transform and reports it as outputs that differ with the MLIR
// detector flagging it.
//
// Expected ground truth: outputs DIFFER (transform fails to apply at run time;
// no transformed kernel is produced).
// Expected detectors: MLIR detected (apply-time interchange/rank mismatch).
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

        // [2, 1, 0] is a permutation of {0,1,2} (passes the static verifier) but
        // has three entries for a 2-D op -> rejected at apply time.
        %fused, %loops:2 = transform.structured.fuse %cons [8, 8] interchange [2, 1, 0]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
