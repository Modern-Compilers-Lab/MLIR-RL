// `transform.structured.tile_reduction_using_forall` (Category A, FP caveat).
//
// The `scf.forall` counterpart of
// tile_reduction_using_for_elementwise_reduction_ok.mlir, on the
// parallel-parallel-reduction shape of tiling_elementwise_reduction_ok.mlir:
// out[d0, d1] = sum_d2 in[d0, d1, d2] * 2. Only the innermost reduction d2 is
// distributed (num_threads = [0, 0, 2]); the two parallel dims are untouched.
//
// `tile_reduction_using_forall` gives each thread a private identity-initialized
// partial, parallel-inserts the per-thread partials into a shared tensor, then a
// final merge reduces them back into `%out`. This is a mathematically valid
// reduction reordering (identity-init + merge): equal in exact arithmetic and
// within tolerance in floating point, so the transformed kernel must MATCH the
// baseline (rtol=1e-5/atol=1e-6). The partials are private, so there is no race.
//
// `%out` is the destination-passing-style accumulator; after bufferization it
// becomes the in-place memref the harness compares. The length-4 reduction is
// evenly divided by the 2 threads, so every thread gets a full 2-element slice.
//
// Expected ground truth: outputs MATCH (within fp tolerance).
// Expected detectors: MLIR silent, Equivalence silent.
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
        %cst = arith.constant 2.0 : f64
        %r = arith.mulf %a, %cst : f64
        %s = arith.addf %acc, %r : f64
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
