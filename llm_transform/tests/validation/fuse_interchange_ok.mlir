// `transform.structured.fuse` with the `interchange` sub-option (Category A
// here; the sub-option is Category C only when it reorders a real dependence).
//
// Same producer/consumer chain as fuse_elementwise_ok.mlir (C = A*2 + B), but
// the tile loops are interchanged to [1, 0] — the d1 tile loop is made outer and
// d0 inner. Both consumer dimensions are "parallel" and carry no cross-iteration
// dependence, so permuting the tile loops is a legal (parallel<->parallel)
// reorder: the structured ops are interchange-invariant and tile-and-fuse still
// recomputes the exact consumed slice.
//
// This is the legal counterpart of the tile-loop-interchange risk: `fuse` with
// `interchange` applies the permutation blindly (MLIR does not check
// dependences), but for a dependence-free elementwise nest the result is
// unchanged, so outputs MATCH and the equivalence verifier — which would flag a
// genuine dependence reversal — stays silent.
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

        %fused, %loops:2 = transform.structured.fuse %cons [8, 8] interchange [1, 0]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
