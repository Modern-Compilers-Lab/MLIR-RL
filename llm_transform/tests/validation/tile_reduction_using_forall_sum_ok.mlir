// `transform.structured.tile_reduction_using_forall` (Category A, FP caveat).
//
// The `scf.forall` counterpart of tile_reduction_using_for_sum_ok.mlir: the
// canonical row-wise sum over an innermost reduction dimension,
// out[d0] = sum_d1 in[d0, d1]. `tile_reduction_using_forall` distributes the
// reduction dimension d1 across `num_threads` threads, gives each thread its
// own identity-initialized partial-sum slice, parallel-inserts the partials
// into a shared tensor, then runs a final merge that reduces the partials back
// into the original output.
//
// This is a mathematically valid reduction reordering (identity-init + merge):
// equal in exact arithmetic and within tolerance in floating point. The harness
// compares with rtol=1e-5/atol=1e-6, so the transformed kernel must MATCH the
// baseline. (The per-thread partials are private, so — unlike a plain
// tile_using_forall on a reduction, see parallel_reduction.mlir — there is no
// race; the construct is correct regardless of thread scheduling.)
//
// `%out` is the destination-passing-style accumulator; after bufferization it
// becomes the in-place memref the harness compares. The length-128 reduction is
// evenly divided by the 8 threads, so every thread gets a full 16-element slice.
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

        %fill, %split, %combine, %forall =
            transform.structured.tile_reduction_using_forall %op by num_threads = [0, 8]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
