// `transform.structured.tile_reduction_using_forall` (Category A, FP caveat).
//
// The `scf.forall` counterpart of tile_reduction_using_for_matmul_ok.mlir, and
// the parallel-reduction sibling of split_reduction_ok.mlir: a matmul whose K
// reduction is distributed across threads. The two parallel dims (M, N) are left
// intact (num_threads 0); only the K dimension is split (num_threads = 4).
//
// Each thread computes a partial-sum matrix into a private identity-initialized
// slice, the per-thread partials are parallel-inserted into a shared tensor, and
// a final merge reduces them back into `%C`. This is a mathematically valid
// reduction reordering (identity-init + merge): equal in exact arithmetic and
// within tolerance in floating point, so the transformed kernel must MATCH the
// baseline (rtol=1e-5/atol=1e-6). Because the partials are private there is no
// race — the result is independent of thread scheduling.
//
// `%C` is the destination-passing-style accumulator (matmul computes C += A·B);
// after bufferization it becomes the in-place memref the harness compares. Both
// variants start from the same random C and the merge preserves it, so the
// totals agree. The K extent 64 is evenly divided by the 4 threads.
//
// Expected ground truth: outputs MATCH (within fp tolerance).
// Expected detectors: MLIR silent, Equivalence silent.
// Harness verdict: [PASS].

func.func @main(%A: tensor<16x64xf64>, %B: tensor<64x16xf64>, %C: tensor<16x16xf64>) -> tensor<16x16xf64> {
    %0 = linalg.matmul {tag = "operation"}
        ins(%A, %B : tensor<16x64xf64>, tensor<64x16xf64>)
        outs(%C : tensor<16x16xf64>) -> tensor<16x16xf64>
    return %0 : tensor<16x16xf64>
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %fill, %split, %combine, %forall =
            transform.structured.tile_reduction_using_forall %op by num_threads = [0, 0, 4]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
