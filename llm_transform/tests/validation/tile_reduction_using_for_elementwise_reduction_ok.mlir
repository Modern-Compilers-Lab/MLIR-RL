// `transform.structured.tile_reduction_using_for` (Category A, FP caveat).
//
// Reduction-tiling analogue of tiling_elementwise_reduction_ok.mlir, but the
// reduction dimension is split with `tile_reduction_using_for` instead of being
// carried along by a plain `tile_using_for`. Each output element is a sum over
// an innermost reduction dimension d2: out[d0, d1] = sum_d2 in[d0, d1, d2] * 2,
// with the two parallel dims left untiled (tile size 0).
//
// `tile_reduction_using_for` accumulates a partial sum per reduction tile into
// an identity-initialized temporary, then a final `for`-carried merge reduces
// the partials back into `%out`. This is a mathematically valid reduction
// reordering (identity-init + merge): equal in exact arithmetic and within
// tolerance in floating point, so the transformed kernel must MATCH the
// baseline (rtol=1e-5/atol=1e-6).
//
// `%out` is the destination-passing-style accumulator; after bufferization it
// becomes the in-place memref the harness compares. The length-4 reduction is a
// multiple of the tile size 2, so only full tiles are produced.
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

        %fill, %split, %combine, %for_op =
            transform.structured.tile_reduction_using_for %op by tile_sizes = [0, 0, 2]
            : (!transform.any_op)
              -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
