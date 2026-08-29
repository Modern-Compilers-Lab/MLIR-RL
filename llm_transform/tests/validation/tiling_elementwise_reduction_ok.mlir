// Reduction-dimension analogue of tiling_elementwise_ok.mlir. Instead of a
// pure elementwise map, each output element is a sum over an innermost
// "reduction" dimension d2: out[d0, d1] = sum_d2 in[d0, d1, d2] * 2. The input
// and output memrefs are disjoint, so there is no cross-iteration dependence.
//
// Tiling the parallel dims [32, 32] is sound for the same reason as the 2D
// sibling, and tiling the reduction dim sequentially (tile size 2) preserves
// the accumulation order, so the transformed code must produce the same result
// as the baseline.

func.func @tiling_elementwise_reduction_ok(%in: memref<128x128x4xf64>, %out: memref<128x128xf64>) {
    linalg.generic {tag = "operation",
        indexing_maps = [
            affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
            affine_map<(d0, d1, d2) -> (d0, d1)>
        ],
        iterator_types = ["parallel", "parallel", "reduction"]
    } ins(%in : memref<128x128x4xf64>)
      outs(%out : memref<128x128xf64>) {
    ^bb0(%a: f64, %acc: f64):
        %cst = arith.constant 2.0 : f64
        %r = arith.mulf %a, %cst : f64
        %s = arith.addf %acc, %r : f64
        linalg.yield %s : f64
    }
    return
}

module attributes {transform.with_named_sequence} {
    transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
        %op = transform.structured.match attributes {tag = "operation"} in %arg0
            : (!transform.any_op) -> !transform.any_op

        %tiled_op, %loops:3 = transform.structured.tile_using_for %op tile_sizes [32, 32, 2]
            : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

        transform.yield
    }
}
