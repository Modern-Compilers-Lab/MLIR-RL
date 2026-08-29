// `transform.structured.tile_reduction_using_for` (Category A, FP caveat).
//
// Reduction-tiling of a matmul's K dimension, the legal sibling of
// split_reduction_ok.mlir. Where `split_reduction` factors K into a parallel +
// reduction pair, `tile_reduction_using_for` strip-mines K into tiles, computes
// a partial-sum matrix per tile into an identity-initialized temporary, then a
// final `for`-carried merge reduces the partials back into `%C`.
//
// The two parallel dims (M, N) are left untiled (tile size 0); only the K
// reduction dim is tiled. This is a mathematically valid reduction reordering
// (identity-init + merge): equal in exact arithmetic and within tolerance in
// floating point, so the transformed kernel must MATCH the baseline
// (rtol=1e-5/atol=1e-6).
//
// `%C` is the destination-passing-style accumulator (matmul computes
// C += A·B); after bufferization it becomes the in-place memref the harness
// compares. Both variants start from the same random C and the merge preserves
// it, so the totals agree. The K extent 64 is a multiple of the tile size 16,
// so only full tiles are produced.
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

        %fill, %split, %combine, %for_op =
            transform.structured.tile_reduction_using_for %op by tile_sizes = [0, 0, 16]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
